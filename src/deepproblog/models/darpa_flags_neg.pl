% Neural networks

nn(net1, [X], Z, [benign, phase1]) :: recon(X, Z).
nn(net2, [X], Z, [benign, phase2]) :: ping(X, Z).
nn(net3, [X], Z, [benign, phase3]) :: overflow(X, Z).
nn(net4, [X], Z, [benign, phase4]) :: install(X, Z).
nn(net5, [X], Z, [benign, phase5]) :: ddos(X, Z).

% Per phase rules

phase(1, X, phase1) :-
    recon(X, phase1).

phase(1, X, benign) :- 
    \+ phase(1, X, phase1).

phase(2, X, phase2) :-
    ping(X, phase2).

phase(2, X, benign) :-
    \+ phase(2, X, phase2).
    
phase(3, X, phase3) :- 
    overflow(X, phase3).

phase(3, X, benign) :-
    \+ phase(3, X, phase3).

phase(4, X, phase4) :-
    install(X, phase4).

phase(4, X, benign) :-
    \+ phase(4, X, phase4).

phase(5, X, phase5) :-
    ddos(X, phase5).

phase(5, X, benign) :-
    \+ phase(5, X, phase5).


% Multi-step attack logic

multi_step(X, Next, Outcome) :-
    phase(Next, X, Outcome). 