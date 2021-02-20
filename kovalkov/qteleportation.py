import qutip as qt
import numpy as np
import itertools


def initialize(state):
    return qt.composite(state, qt.bell_state(state='00'))


def evolute(state):
    return qt.snot(N=3, target=0)*qt.cnot(N=3, control=0, target=1) * state


def measure(state):
    confs = list(itertools.product([0, 1], repeat=2))
    probabilities = []
    for m0, m1 in confs:
        P = qt.tensor([
            qt.basis(2, m0).proj(),
            qt.basis(2, m1).proj(),
            qt.Qobj(np.eye(2))])
        probabilities.append(np.real((state.dag() * P * state).full()[0][0]))
    return np.random.choice([0, 1, 2, 3], p=probabilities)


def teleport(state, mres):
    X_ = 1/(-1j)*qt.rx(np.pi, N=3, target=2)
    Z_ = 1/(-1j)*qt.rz(np.pi, N=3, target=2)
    if mres == 1:
        state = X_ * state
    if mres == 2:
        state = Z_ * state
    if mres == 3:
        state = Z_ * X_ * state
    state_array = state.full()
    amp = [state_array[0], state_array[1]]
    return qt.Qobj(amp).unit()



if __name__=='__main__':
    state = qt.rand_ket(N=2)
    result = evolute(initialize(state))
    teleport_state = teleport(result, measure(result))
    print(state.full())
    print(teleport_state.full())
