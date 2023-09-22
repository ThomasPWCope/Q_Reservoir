# -*- coding: utf-8 -*-
"""Contains :class:`QRewindingRC` and :class:`QRewindingStatevectorRC` classes.

:class:`QRewindingStatevectorRC` is the quantum reservoir model performing the best.
In theory :class:`QRewindingRC` should perform the same, but a lot slower.
In practice :class:`QRewindingStatevectorRC` is has been much more tested and is highly recommended in all situations.
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeManila, FakeToronto, FakeJakartaV2, FakeProvider
from qiskit_aer.noise import NoiseModel
import qiskit.quantum_info as qi
import qiskit_aer
import qiskit

from qiskit.quantum_info.operators.predicates import is_hermitian_matrix, is_positive_semidefinite_matrix

import numpy as np
import sys
import os

sys.path.insert(1, os.path.abspath('..')) # module: qrc_surrogate

from src.continuous import QContinuousRC
from src.basemodel import PredictionModelBase, QuantumBase, StepwiseModelBase
from src.data import DataSource
from src.helpers import mse, mape, nrmse, moving_average, weighted_gate_count, unnormalize

# np.set_printoptions(precision=2, suppress=True)

step_to_save_qc = 10

class QRewindingRC(StepwiseModelBase, QContinuousRC):
    """
    Takes x(t) (action) and previous ouput y(t-1) as input.
    Repeats past n steps and the current step, then measures all qubits.

    During training the model takes all x(t) as given and uses the correct y(t-1) from the data.
    During inference (validating) uses the previous predicted output ~ y(t-1) and x(t) can depend on the previous prediction.
    
    Args:
        nmeas (int): 
            Number of qubits to be measured at every rewinded (non-final) step.
            A final step all qubits are measured.
        reset_instead_meas (bool):
            Reset qubits instead of measuring at every rewinded (non-final) step.
        resetm (bool):
            Reset qubits after measurement at every rewinded (non-final) step.
        use_partial_meas (bool):
            If partial measurements at every rewinded (non-final) step are to be used for features.
            Set to 'False' if 'reset_instead_meas' is 'True'.
        lookback (int):
            Number of steps to be rewinded at every step.
        restarting (bool):
            If to restart at every step (rewind to the beginning).
        lookback_max (bool):
            If restarting is True, this is ignored (and set to True).
            If True, the number of steps to be rewinded is the maximum possible (all previous steps),
            but never goes further back than step 0.
            If False, steps before t=0 are possibly repeated as well, with input values depending on :code:`set_past_y_to_0`.
        add_y_to_input (bool):
            If to use previous y(t) in addition to x(t) as input.
    
    Additional arguments are passed to :class:`src.basemodel.StepwiseModelBase`
    and :class:`src.continuous.QOnlineReservoir`.
    """
    model_name = 'rewinding_rc'

    def __init__(
        self,
        # QRewindingRC
        use_partial_meas = False,
        reset_instead_meas = True,
        lookback = 3,
        lookback_max = True,
        restarting = False,
        mend = False,
        add_y_to_input = True,
        # -
        # QOnlineReservoir
        washout = 0, # number of first steps of episode ignored by fitting/error metrics
        preloading = 0, # number of times the first step is repeated before episode actually starts
        mtype = 'projection',
        minc = True, 
        nmeas = 1, # number of measured qubits
        reseti = True, # reset before input
        resetm = False, # reset after measurements
        nenccopies = 1,
        # -
        # StepwiseModelBase
        xyoffset = 1,
        set_past_y_to_0 = True,
        use_true_y_in_val = False,
        # -
        # PredictionModelBase
        rseed = 0,
        log = True,
        add_x_as_feature = True,
        # predicting multiple steps forward
        nyfuture = 1, 
        delete_future_y = True,
        # fitter
        fitter = 'sklearn',
        regression_model = 'regression',
        regression_alpha = 0.1,
        regression_l1 = 0.1,
        poly_degree = 3,
        # -
        # QuantumBase
        nqubits = 5,
        qctype = 'ising',
        qinit = 'none',
        nlayers = 1, # unitaries per timestep
        ftype = 0,
        enctype = 'angle',
        encaxes = 1, # number of axis for encoding
        measaxes = 3,
        encangle = 1, # if = 1, encoding is single qubit rotation from 0 to 1*Pi
        shots = 2**13, # 8192
        # ising
        ising_t = 1,
        ising_jmax = 1,
        ising_h = .1,
        ising_wmax = 10,
        ising_random = True,
        ising_jpositive = False,
        ising_wpositive = False,
        # sim
        sim = 'aer_simulator',
        t1 = 50,
        sim_method = 'statevector',
        sim_precision = 'single',
    ) -> None:
        QContinuousRC.__init__(
            self,
            washout = washout,
            preloading = preloading,
            mtype = mtype,
            minc = minc, 
            mend = mend,
            nmeas = nmeas,
            reseti = reseti, 
            resetm = resetm, 
            nenccopies = nenccopies,
            # -
            # PredictionModelBase
            rseed = rseed,
            log = log,
            add_x_as_feature = add_x_as_feature,
            # predicting multiple steps forward
            nyfuture = nyfuture, 
            delete_future_y = delete_future_y,
            # fitter
            fitter = fitter,
            regression_model = regression_model,
            regression_alpha = regression_alpha,
            regression_l1 = regression_l1,
            poly_degree = poly_degree,
            # -
            # QuantumBase
            nqubits = nqubits,
            qctype = qctype,
            qinit = qinit,
            nlayers = nlayers, 
            ftype = ftype,
            enctype = enctype,
            measaxes = measaxes,
            encaxes = encaxes,
            encangle = encangle,
            shots = shots, # 8192
            # ising
            ising_t = ising_t,
            ising_jmax = ising_jmax,
            ising_h = ising_h,
            ising_wmax = ising_wmax,
            ising_random = ising_random,
            ising_jpositive = ising_jpositive,
            ising_wpositive = ising_wpositive,
            # sim
            sim = sim,
            t1 = t1,
            sim_method = sim_method,
            sim_precision = sim_precision,
        )
        StepwiseModelBase.__init__(
            self,
            xyoffset = xyoffset,
            set_past_y_to_0 = set_past_y_to_0,
            use_true_y_in_val = use_true_y_in_val,
            lookback_max = lookback_max
        )
        # partial measurement related
        self.reset_instead_meas = reset_instead_meas
        self.use_partial_meas = use_partial_meas 
        if reset_instead_meas == True:
            if self.resetm == False:
                print('! reset_instead_meas is True, setting resetm to True !')
            self.resetm = True
        if restarting == True:
            self.lookback_max = True
        if restarting == True or self.lookback_max == True:
            assert self.use_partial_meas == False, f"""
                If restarting is True, use_partial_meas must be False.
                Reason: there would be a different number of measurements, 
                and thus number of features, at each step, 
                which cannot be handled by the OLS fitter."""
        # 
        lookback = max(1, lookback)
        self.restarting = restarting 
        if restarting == True:
            lookback = 1
        self.lookback = lookback
        self.ylookback = lookback 
        self.xlookback = lookback
        self.add_y_to_input = add_y_to_input
        if self.add_y_to_input == False:
            self.ylookback = 0
        return

    def _set_unitary_and_meas(self):
        """Define unitary quantum circuit at every timestep.
        Sets self.unistep.
        Excludes input encoding and measurements.
        """
        self.dimx_wo_copies = self.dimx
        if self.ylookback > 0:
            self.dimx_wo_copies += self.dimy
        self.dimxqc = int(self.dimx_wo_copies * self.nenccopies)
        assert self.dimxqc <= self.nqubits, f'{self.dimxqc} {self.nqubits}'
        # dimension of the input data at each step
        if self.ylookback > 0:
            self.dmin = np.asarray([self.xmin, self.ymin]).reshape(-1) # (dimxqc,)
            self.dmax = np.asarray([self.xmax, self.ymax]).reshape(-1) # (dimxqc,)
        else:
            self.dmin = np.asarray(self.xmin).reshape(-1) # (dimxqc,)
            self.dmax = np.asarray(self.xmax).reshape(-1) # (dimxqc,)
        self.qin = [*range(self.dimxqc)]
        self.quni = [*range(self.nqubits)]
        if self.mend:
            self.qmeas = [*range(self.nqubits-self.nmeas, self.nqubits)]
        else:
            self.qmeas = [*range(self.nmeas)]
        self.qreset = []
        if self.resetm == True:
            self.qreset += self.qmeas
        if self.reseti == True:
            self.qreset += self.qin
        self.qreset = list(set(self.qreset))
        # classical bits for measuring
        self.ncbits = self.nqubits
        if self.use_partial_meas:
            self.ncbits += (self.nmeas * self.lookback)
            self.cbits_final_meas = [*range(self.ncbits-self.nqubits, self.ncbits)]
        else:
            self.cbits_final_meas = self.quni
        if self.unistep is None:
            self._set_unitary()
            # number of gates for one timestep without input encoding (reservoir)
            self.unistep.name = 'U'
            qct = transpile(self.unistep, backend=self.backend)
            self.ngates = weighted_gate_count(qct)
        else:
            print('! Unitary is already defined !')
        return
    
    def _get_step_features(self, angles, nsteps=1, saveqc=False):
        """Features for this step, provided the input angles.
        At every step, the circuit consists of lookback+1 steps
        
        angles (np.ndarray): (nsteps, dimxqc)
        nsteps (int): how many previous steps should be repeated (including the current step).
        
        Return:
            features for this step (1, dimf)
        """
        features_step = []
        for nax, ax in enumerate(self.measaxes):
            qc = QuantumCircuit(self.nqubits, self.ncbits)
            if self.qinit == 'h':
                qc.h(self.quni)
            for prevstep in range(nsteps-1):
                # input
                self._add_input_to_qc(qc=qc, angles=angles, step=prevstep)
                # unitary
                qc.append(self.unistep, self.quni)
                if self.nmeas > 0:
                    if self.reset_instead_meas:
                        qc.reset(self.qmeas)
                    elif self.use_partial_meas:
                        match ax:
                            # https://arxiv.org/abs/1804.03719
                            case 'z':
                                pass
                            case 'x':
                                qc.h(self.qmeas)
                            case 'y':
                                qc.sdg(self.qmeas)
                                qc.h(self.qmeas)
                            case _:
                                raise Warning(f'Invalid measaxes {self.measaxes}')
                        qc.measure(
                            qubit=self.qmeas, 
                            cbit=[*range(prevstep*self.nmeas, (prevstep+1)*self.nmeas)]
                        )
                    else:
                        # the cbits will be overwritten at every step, only the last one will be kept
                        qc.measure(qubit=self.qmeas, cbit=[*range(self.nmeas)])
            # final step
            # input
            self._add_input_to_qc(qc=qc, angles=angles, step=nsteps-1)
            # unitary
            qc.append(self.unistep, self.quni)
            # measure
            match ax:
                # https://arxiv.org/abs/1804.03719
                case 'z':
                    pass
                case 'x':
                    qc.h(self.quni)
                case 'y':
                    qc.sdg(self.quni)
                    qc.h(self.quni)
                case _:
                    raise Warning(f'Invalid measaxes {self.measaxes}')
            qc.measure(qubit=self.quni, cbit=self.cbits_final_meas) 
            if saveqc:
                self.qc = qc
            compiled_qc = transpile(qc, self.backend)
            job = self.backend.run(compiled_qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            # qiskit counts are in reverse order
            counts = {k[::-1]: v for k, v in counts.items()}
            # turn measurements into features
            if self.use_partial_meas:
                for prev_step in range(nsteps-1):
                    counts_step = {}
                    b_step = b[prev_step * self.nmeas:(prev_step + 1) * self.nmeas]
                    for b, c in counts.items():
                        if b_step in counts_step.keys():
                            counts_step[b_step] += c
                        else:
                            counts_step[b_step] = c
                    features_step = self._step_meas_to_step_features(counts_step, features_step)
                # final step
                counts_final = {}
                b_final = b[len(self.quni):]
                for b, c in counts.items():
                    if b_step in counts_final.keys():
                        counts_final[b_final] += c
                    else:
                        counts_final[b_final] = c
                features_step = self._step_meas_to_step_features(counts_final, features_step)
            else:
                features_step = self._step_meas_to_step_features(counts_step=counts, features_step=features_step)
        return features_step

    def _get_input_t(self, xe_lookback, ye_lookback, t_in, t_pred):
        """
        Every step is a mini-episode:
        At every step, repeat past n steps + current step. 
        At the current step, feed in current action x(t) and previous ouput y(t-1).
        At the past n'th step, feed in action x(t-n) and previous output y(t-n-1).
        In training previous outpus y are the true values from the data.
        
        Returns:
            angles (np.ndarray): (lookback, dimxqc)
        """
        if self.restarting:
            return np.hstack([
                xe_lookback[0:t_in+1],
                ye_lookback[0:t_in+1],
            ])
        else:
            if self.lookback_max:
                if self.ylookback > 0:
                    # t_pred = 4, t_in = 3, lookback = 3 -> 1, 2, 3
                    return np.hstack([
                        xe_lookback[max(0, t_in-self.ylookback+1):t_in+1],
                        ye_lookback[max(0, t_in-self.ylookback+1):t_in+1], 
                    ])
                else:
                    return xe_lookback[max(0, t_in-self.xlookback+1):t_in+1]
            else:
                if self.ylookback > 0:
                    return np.hstack([
                        xe_lookback[t_in:t_in+self.xlookback],
                        ye_lookback[t_in:t_in+self.ylookback], 
                    ])
                else:
                    return xe_lookback[t_in:t_in+self.xlookback]
    
    def _t_input_to_t_features(self, input_t, x0, t_pred):
        """Features of a step, given the input.
        
        Calls angle_encoding and get_step_features.
        """
        # encode input for quantum circuit
        step_input_angles = self._angle_encoding(episode=input_t, dmin=self.dmin, dmax=self.dmax)
        # run circuit, get features
        if self.lookback_max:
            lookback = min(self.xlookback, t_pred)
        else:
            lookback = self.xlookback
        # print(f'lookback {lookback} t_pred {t_pred}')
        features_qc = self._get_step_features(
            angles=step_input_angles, 
            nsteps=t_pred if self.restarting == True else lookback,
            saveqc=True if t_pred == step_to_save_qc else False,
        )
        if self.add_x_as_feature:
            return np.hstack(features_qc + [x0]) 
        else:
            return np.hstack(features_qc)
        


class QRewindingStatevectorRC(QRewindingRC):
    """
    Statevector.

    Intended to be faster when using mid circuit measurements or resets.

    If set to restarting, will evolve one statevector through all steps.
    If set to rewinding, will create new statevector and evolve through past steps.

    When doing noisy simulations we need to move from Statevector to DensityMatrix.
    Considerably slower. 

    Args:
        sim_sampling (str): 
            | method to sample counts from statevector, which are then used to estimate expectation values.
            | qiskit: use qiskit's :code:`qiskit.quantum_info.Statevector.sample_counts()`. crazy slow.
            | multinomial: :code:`np.multinomial`. 10x faster and same result as qiskit.
                Speedup is less when compared to :code:`qiskit.quantum_info.DensityMatrix.sample_counts()`.
            | naive: uses exact probabilies to generate counts. expectation values are still estimated from counts.
            | exact: calculates expectation values directly from statevector.

    Additional arguments are passed to :class:`.QRewindingRC`.
    """
    model_name = 'rewinding_statevector_rc'

    def __init__(
        self,
        # QRewindingStatevectorRC
        sim_sampling = 'multinomial',
        # QRewindingRC
        use_partial_meas = False,
        reset_instead_meas = False,
        lookback = 3,
        add_y_to_input = True,
        restarting = False,
        mend = False,
        set_past_y_to_0 = True,
        use_true_y_in_val = False,
        # -
        # QOnlineReservoir
        washout = 0, # number of first steps of episode ignored by fitting/error metrics
        preloading = 0, # number of times the first step is repeated before episode actually starts
        mtype = 'projection',
        minc = True, 
        nmeas = 1, # number of measured qubits
        reseti = True, # reset before input
        resetm = True, # reset after measurements
        # -
        lookback_max = True,
        # PredictionModelBase
        rseed = 0,
        log = True,
        add_x_as_feature = True,
        # predicting multiple steps forward
        nyfuture = 1, 
        delete_future_y = True,
        # fitter
        fitter = 'sklearn',
        regression_model = 'regression',
        regression_alpha = 0.1,
        regression_l1 = 0.1,
        poly_degree = 3,
        # -
        # QuantumBase
        nqubits = 5,
        qctype = 'ising',
        qinit = 'none',
        nlayers = 1, # unitaries per timestep
        ftype = 0,
        enctype = 'angle',
        encaxes = 1,
        nenccopies = 1,
        encangle = 1, # if = 1, encoding is single qubit rotation from 0 to 1*Pi
        measaxes = 3, # number of axis for measurements
        shots = 2**13, # 8192
        # ising
        ising_t = 1,
        ising_jmax = 1,
        ising_h = .1,
        ising_wmax = 10,
        ising_random = True,
        ising_jpositive = False,
        ising_wpositive = False,
        # sim
        sim = 'aer_simulator',
        t1 = 50,
        sim_method = 'statevector',
        sim_precision = 'double',
    ) -> None:
        QRewindingRC.__init__(
            self,
            # QRewindingRC
            use_partial_meas = use_partial_meas,
            reset_instead_meas = reset_instead_meas,
            lookback = lookback,
            lookback_max = lookback_max,
            restarting = restarting,
            mend = mend,
            set_past_y_to_0 = set_past_y_to_0,
            add_y_to_input = add_y_to_input,
            use_true_y_in_val = use_true_y_in_val,
            # -
            # QOnlineReservoir
            washout = washout,
            preloading = preloading,
            mtype = mtype,
            minc = minc, 
            nmeas = nmeas,
            reseti = reseti, 
            resetm = resetm, 
            # -
            # PredictionModelBase
            rseed = rseed,
            log = log,
            add_x_as_feature = add_x_as_feature,
            # predicting multiple steps forward
            nyfuture = nyfuture, 
            delete_future_y = delete_future_y,
            # fitter
            fitter = fitter,
            regression_model = regression_model,
            regression_alpha = regression_alpha,
            regression_l1 = regression_l1,
            poly_degree = poly_degree,
            # -
            # QuantumBase
            nqubits = nqubits,
            qctype = qctype,
            qinit = qinit,
            nlayers = nlayers, 
            ftype = ftype,
            enctype = enctype,
            encaxes = encaxes,
            nenccopies = nenccopies,
            encangle = encangle,
            measaxes = measaxes, 
            shots = shots, # 8192
            # ising
            ising_t = ising_t,
            ising_jmax = ising_jmax,
            ising_h = ising_h,
            ising_wmax = ising_wmax,
            ising_random = ising_random,
            ising_jpositive = ising_jpositive,
            ising_wpositive = ising_wpositive,
            # sim
            sim = sim,
            t1 = t1,
            sim_method = sim_method,
            sim_precision = sim_precision,
        )
        # - Set simulator
        self.sim_sampling = sim_sampling
        if self.sim_sampling == 'naive':
            self.shots = 2**13
        if self.sim_method == 'density_matrix':
            self.dm = True
            if self.sim_precision == 'single':
                print("! Setting sim_precision='single' with sim_method='density_matrix' can lead to dm.trace!=1.")
        elif self.sim_method == 'statevector':
            self.dm = False
        # Set statevector
        # used for restarting
        self._init_statevector()
        # - 
        if resetm == False:
            raise NotImplementedError('Only resetm=True (partial tracing the statevector after getting the counts) has been implemented')
        self.noisy_dm = False
        if self.backend.options.noise_model != None:
            self.noisy_dm = True
            if self.sim_method != 'density_matrix':
                raise NotImplementedError(
                    f"""Noise models for Statevector hasnt been implemented: 
                    {self.backend.options.noise_model}, {self.sim_method}.
                    Set sim_method='density_matrix'."""
                )
    
    def _init_statevector(self):
        self.statev = qi.Statevector.from_label('0'*self.nqubits) # difference SV
        if self.qinit == 'h':
            self.statev = qi.Statevector.from_label('+'*self.nqubits)
        if self.dm:
            self.statev = qi.DensityMatrix(self.statev.data)
        return
    
    def _get_input_t(self, xe_lookback, ye_lookback, t_in, t_pred=None):
        """
        Every step is a mini-episode:
        At every step, repeat past n steps + current step. 
        At the current step, feed in current action x(t) and previous ouput y(t-1).
        At the past n'th step, feed in action x(t-n) and previous output y(t-n-1).
        In training previous outpus y are the true values from the data.
        
        Returns:
            angles (np.ndarray): (lookback, dimxqc)
        """
        if t_in == 0:
            self._init_statevector()
        # if restarting is True, will return one step at a time
        if self.lookback_max:
            if self.ylookback > 0:
                # t_pred = 4, t_in = 3, lookback = 3 -> 1, 2, 3
                return np.hstack([
                    xe_lookback[max(0, t_in-self.ylookback+1):t_in+1],
                    ye_lookback[max(0, t_in-self.ylookback+1):t_in+1], 
                ])
            else:
                return xe_lookback[max(0, t_in-self.xlookback+1):t_in+1]
        else:
            if self.ylookback > 0:
                return np.hstack([
                    xe_lookback[t_in:t_in+self.xlookback],
                    ye_lookback[t_in:t_in+self.ylookback], 
                ])
            else:
                return xe_lookback[t_in:t_in+self.xlookback]
    
    def _t_input_to_t_features(self, input_t, x0, t_pred):
        """Features of a step, given the input.
        
        Calls angle_encoding and get_step_features.
        """
        # encode input for quantum circuit
        step_input_angles = self._angle_encoding(episode=input_t, dmin=self.dmin, dmax=self.dmax)
        # run circuit, get features
        if self.lookback_max: 
            lookback = min(self.xlookback, t_pred)
        else:
            lookback = self.xlookback
        features_qc = self._get_step_features(
            angles=step_input_angles, 
            nsteps=lookback, 
            saveqc=True if t_pred == step_to_save_qc else False,
        )
        if self.add_x_as_feature:
            return np.hstack(features_qc + [x0]) 
        else:
            return np.hstack(features_qc)
    
    def _get_step_features(self, angles, nsteps=1, saveqc=False):
        """Features for this step. 
        At every step, the circuit consists of lookback+1 steps.
        statev: qiskit.quantum_info.Statevector() or qiskit.quantum_info.DensityMatrix()
        
        angles (np.ndarray): (nsteps, dimxqc)
        nsteps (int): how many previous steps should be repeated (including the current step).
            
        
        Returns:
            features for this step (1, dimf)
        """
        features_step = []
        if self.restarting == False: 
            self._init_statevector()
        assert np.shape(angles)[0] == nsteps, f'{np.shape(angles)}[0] == {nsteps}'
        for prevstep in range(nsteps-1):
            if self.restarting == True: 
                raise ValueError('Shouldnt be here')
            qc_prevsteps = QuantumCircuit(self.nqubits)
            if self.noisy_dm == True: # set get dm
                # self.statev = _allow_invalid_dm(self.statev)
                qc_prevsteps.set_density_matrix(self.statev)
            # input
            self._add_input_to_qc(qc=qc_prevsteps, angles=angles, step=prevstep)
            # unitary
            qc_prevsteps.append(self.unistep, self.quni)
            # run qc
            if self.noisy_dm == True: 
                # get dm
                qc_prevsteps.save_density_matrix(qubits=None, label="dm", conditional=False)
                qc_prevsteps = transpile(qc_prevsteps, self.backend)
                job = self.backend.run(qc_prevsteps)
                result = job.result()
                self.statev = result.data()['dm']
            else:
                # evolve
                self.statev = self.statev.evolve(qc_prevsteps)
            if (self.nmeas > 0) and (self.use_partial_meas == True):
                # get measurements for all axis
                # instead Statevector.expectation_value(oper=, qargs=)
                for nax, ax in enumerate(self.measaxes): 
                    statev_ax = self._get_rotated_sv(ax=ax, nqubits=qc_prevsteps.num_qubits, qargs=self.qmeas)
                    if self.sim_sampling == 'exact':
                        self._step_meas_to_step_features_sv(sv=statev_ax, features_step=features_step)
                    else:
                        counts_step = self._counts_from_sv(sv=statev_ax)
                        # counts_step = statev_ax.sample_counts(qargs=self.qmeas, shots=self.shots)
                        features_step = self._step_meas_to_step_features(counts_step, features_step)
            if len(self.qreset) > 0:
                self._reset_sv()
        # del qc_prevsteps
        # final step
        qc_final = QuantumCircuit(self.nqubits)
        if self.noisy_dm == True: # set get dm
            # self.statev = _allow_invalid_dm(self.statev)
            qc_final.set_density_matrix(self.statev)
        # input
        # if nsteps-1 >= 0
        self._add_input_to_qc(qc=qc_final, angles=angles, step=nsteps-1)
        # unitary
        qc_final.append(self.unistep, self.quni)
        # run qc
        if self.noisy_dm == True:
            qc_final.save_density_matrix(qubits=None, label="dm", conditional=False)
            qc_final = transpile(qc_final, self.backend)
            job = self.backend.run(qc_final)
            result = job.result()
            self.statev = result.data()['dm']
        else:
            self.statev = self.statev.evolve(qc_final)
        if saveqc:
            self.qc = qc_final
        # measure
        for nax, ax in enumerate(self.measaxes):
            statev_ax = self._get_rotated_sv(ax=ax, nqubits=qc_final.num_qubits, qargs=self.quni)
            if self.sim_sampling == 'exact':
                self._step_meas_to_step_features_sv(sv=statev_ax, features_step=features_step)
            else:
                # get counts
                counts_final = self._counts_from_sv(sv=statev_ax)
                # turn measurements into features
                features_step = self._step_meas_to_step_features(counts_step=counts_final, features_step=features_step)
        if self.restarting == True:  
            if len(self.qreset) > 0:
                self._reset_sv() 
        return features_step
    
    def _get_rotated_sv(self, ax, nqubits, qargs) -> qiskit.quantum_info.Statevector:
        qc_ax = QuantumCircuit(nqubits)
        if self.noisy_dm == True:
            # self.statev = _allow_invalid_dm(self.statev)
            qc_ax.set_density_matrix(self.statev)
        match ax:
            # https://arxiv.org/abs/1804.03719
            case 'z':
                pass
            case 'x':
                qc_ax.h(qargs)
            case 'y':
                qc_ax.sdg(qargs)
                qc_ax.h(qargs)
            case _:
                raise Warning(f'Invalid measaxes {self.measaxes}')
        if self.noisy_dm == True:
            qc_ax.save_density_matrix(qubits=None, label="dm", conditional=False)
            qc_ax = transpile(qc_ax, self.backend)
            job = self.backend.run(qc_ax)
            result = job.result()
            statev_ax = result.data()['dm']
        else:
            statev_ax = self.statev.evolve(qc_ax)
        return statev_ax
    
    def _reset_sv(self) -> qiskit.quantum_info.Statevector:
        # reset measured qubits
        if self.resetm or self.reset_instead_meas:
            statev_meas = qi.partial_trace(self.statev, qargs=self.qreset)
            statev_reset = qi.Statevector.from_label('0'*len(self.qreset))
            if self.dm:
                statev_reset = qi.DensityMatrix(statev_reset.data)
            if self.mend:
                self.statev = statev_reset.tensor(statev_meas)
            else:
                self.statev = statev_meas.tensor(statev_reset)
        else:
            # project statevector instead of resetting
            # self.statev = self.statev.measure(shots=1)
            raise NotImplementedError()
        return

    def _counts_from_sv(self, sv):
        """Naive alternative to Qiskit's statevector.get_counts() using probabilities.
        
        Qiskit becomes very slow for growing number of qubits. 
        I think Qiskit uses a metropolis hasting (Monte Carlo) algorithm under the hood.

        Args:
            sv (qiskit.quantum_info.statevector)
        
        Returns:
            counts (dict)
        """
        if self.sim_sampling == 'qiskit':
            return sv.sample_counts(qargs=self.quni, shots=self.shots)
        elif self.sim_sampling == 'inverse':
            raise NotImplementedError(f"sim_sampling '{self.sim_sampling}' is not implemented in 'counts_from_sv'.")
        elif self.sim_sampling == 'multinomial':
            probs = sv.probabilities() # (2^N,)
            counts_mn = np.random.multinomial(n=self.shots, pvals=probs) 
            return {format(i, "b").zfill(sv.num_qubits): c for i, c in enumerate(counts_mn)} 
            # return {format(i, "b").zfill(sv.num_qubits)[::-1]: c for i, c in enumerate(counts_mn)} 
        elif self.sim_sampling == 'naive':
            return sv.probabilities_dict()
            # return {key: p*self.shots for key, p in probs.items()}
        else:
            raise NotImplementedError(f"sim_sampling '{self.sim_sampling}' is not implemented in 'counts_from_sv'.")


def _allow_invalid_dm(sv):
    # https://github.com/Qiskit/qiskit-terra/blob/26886a1b2926a474ea06aa3f9ce9e11e6ce28020/qiskit/quantum_info/states/densitymatrix.py#L188
    # https://github.com/Qiskit/qiskit-terra/blob/26886a1b2926a474ea06aa3f9ce9e11e6ce28020/qiskit/quantum_info/operators/mixins/tolerances.py#L22
    # default: atol 1e-8, rtol 1e-5
    # sv._RTOL_DEFAULT = 1e-3 # doesnt change anything
    # sv._ATOL_DEFAULT = 1e-5
    sv.__class__._MAX_TOL = 1e-3
    sv.__class__.rtol = 1e-3
    sv.__class__.atol = 1e-5
    # assert isinstance(sv, qi.DensityMatrix), f'{type(sv)}'
    # assert sv.is_valid(), f'trace {sv.trace()}, hermitian {is_hermitian_matrix(sv.data)}, positive semidefinite {is_positive_semidefinite_matrix(sv.data)}'
    return sv