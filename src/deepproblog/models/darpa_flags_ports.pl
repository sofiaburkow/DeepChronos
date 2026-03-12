% Neural networks

nn(net1, [X], Z, [benign, phase1]) :: recon(X, Z).
nn(net2, [X], Z, [benign, phase2]) :: ping(X, Z).
nn(net3, [X], Z, [benign, phase3]) :: overflow(X, Z).
nn(net4, [X], Z, [benign, phase4]) :: install(X, Z).
nn(net5, [X], Z, [benign, phase5]) :: ddos(X, Z).

% Phase specific rules

t(1.0) :: sadmind_port(111).
t(0.9) :: sadmind_port(Port) :- Port >= 32771.
t(0.7) :: sadmind_port(23).
t(0.1) :: sadmind_port(_).

phase(1, X, _, Outcome) :- 
    recon(X, Outcome).

phase(2, X, VictimPort, Outcome) :- 
    sadmind_port(VictimPort),
    phase(1, X, Outcome).

phase(3, X, VictimPort, Outcome) :- 
    sadmind_port(VictimPort),
    phase(2, X, VictimPort, Outcome).

phase(4, X, VictimPort, Outcome) :-
    install(X, Outcome).

phase(5, X, VictimPort, Outcome) :-
    ddos(X, Outcome).

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

multi_step(X, P1, P2, P3, P4, Evidence, DPort, Outcome) :-
    next_phase(P1, P2, P3, P4, Next),
    phase(Next, X, DPort, Outcome),
    bucket(Evidence, B), 
    support_level(B).