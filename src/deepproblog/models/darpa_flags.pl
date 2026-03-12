% Neural networks for each phase
nn(net1, [X], Z, [benign, phase1]) :: phase(1, X, Z).
nn(net2, [X], Z, [benign, phase2]) :: phase(2, X, Z).
nn(net3, [X], Z, [benign, phase3]) :: phase(3, X, Z).
nn(net4, [X], Z, [benign, phase4]) :: phase(4, X, Z).
nn(net5, [X], Z, [benign, phase5]) :: phase(5, X, Z).

% Evidence based confidence

t(0.2) :: support_level(0).
t(0.7) :: support_level(1).
t(0.9) :: support_level(2).
t(0.97) :: support_level(3).

bucket(C,B) :-
    C >= 3, B = 3.

bucket(C,C) :-
    C < 3.

% Multi-step attack reasoning

next_phase(P1, P2, P3, P4, Next) :- 
    Next is P1 + P2 + P3 + P4 + 1.

multi_step(X, P1, P2, P3, P4, Evidence, Outcome) :-
    next_phase(P1, P2, P3, P4, Next),
    phase(Next, X, Outcome),
    bucket(Evidence, B), 
    support_level(B).