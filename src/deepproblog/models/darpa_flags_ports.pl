% Neural networks

nn(net1, [X], Z, [benign, phase1]) :: recon(X, Z).
nn(net2, [X], Z, [benign, phase2]) :: ping(X, Z).
nn(net3, [X], Z, [benign, phase3]) :: overflow(X, Z).
nn(net4, [X], Z, [benign, phase4]) :: install(X, Z).
nn(net5, [X], Z, [benign, phase5]) :: ddos(X, Z).

% Vulnerability knowledge

sadmind_known_port(111).
sadmind_known_port(Port) :- Port >= 32771.
sadmind_known_port(23).

sadmind_port(P) :- sadmind_known_port(P).

% Phase specific rules

phase(1, X, _, Outcome) :- 
    recon(X, Outcome).

phase(2, X, VictimPort, phase2) :-
    sadmind_port(VictimPort),
    ping(X, phase2).

phase(2, X, _, benign) :-
    ping(X, benign).

phase(3, X, VictimPort, Outcome) :- 
    overflow(X, Outcome).

phase(4, X, _, Outcome) :-
    install(X, Outcome).

phase(5, X, _, Outcome) :-
    ddos(X, Outcome).


% Evidence based confidence

0.20 :: support_level(0).
0.60 :: support_level(1).
0.95 :: support_level(2).


% Multi-step attack reasoning

multi_step(X, Next, Evidence, DPort, Outcome) :-
    phase(Next, X, DPort, Outcome),
    support_level(Evidence).