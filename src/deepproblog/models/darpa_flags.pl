% Neural networks for each phase
nn(net1, [X], Z, [benign, phase1]) :: phase(1, X, Z).
nn(net2, [X], Z, [benign, phase2]) :: phase(2, X, Z).
nn(net3, [X], Z, [benign, phase3]) :: phase(3, X, Z).
nn(net4, [X], Z, [benign, phase4]) :: phase(4, X, Z).
nn(net5, [X], Z, [benign, phase5]) :: phase(5, X, Z).

% Evidence based confidence
t(0.3) :: support_level(0).
t(0.70) :: support_level(1).
t(0.95) :: support_level(2).

% Multi-step attack reasoning
multi_step(X, Next, Evidence, Outcome) :-
    phase(Next, X, Outcome),
    support_level(Evidence).