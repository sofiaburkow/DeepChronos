% Neural networks

nn(net1, [X], Z, [benign, phase1]) :: phase(1, X, Z).
nn(net2, [X], Z, [benign, phase2]) :: phase(2, X, Z).
nn(net3, [X], Z, [benign, phase3]) :: phase(3, X, Z).
nn(net4, [X], Z, [benign, phase4]) :: phase(4, X, Z).
nn(net5, [X], Z, [benign, phase5]) :: phase(5, X, Z).


% Multi-step attack logic

multi_step(X, Next, _, _, _, _, _, _, Outcome) :-
    phase(Next, X, Outcome).