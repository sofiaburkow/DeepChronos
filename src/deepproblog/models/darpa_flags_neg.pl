% Neural networks

nn(net1, [X], Z, [benign, attack]) :: ping(X, Z).
nn(net2, [X], Z, [benign, attack]) :: probing(X, Z). 
nn(net3, [X], Z, [benign, attack]) :: exploit(X, Z).
nn(net4, [X], Z, [benign, attack]) :: install(X, Z).
nn(net5, [X], Z, [benign, attack]) :: ddos(X, Z).

% Per phase rules

phase(1, X, phase1) :-
    ping(X, attack).

phase(1, X, benign) :- 
    \+ phase(1, X, phase1).

phase(2, X, phase2) :-
    probing(X, attack).
    
phase(2, X, benign) :-
    \+ phase(2, X, phase2).
    
phase(3, X, phase3) :- 
    exploit(X, attack).

phase(3, X, benign) :-
    \+ phase(3, X, phase3).

phase(4, X, phase4) :-
    install(X, attack).

phase(4, X, benign) :-
    \+ phase(4, X, phase4).

phase(5, X, phase5) :-
    ddos(X, attack).

phase(5, X, benign) :-
    \+ phase(5, X, phase5).


% Multi-step attack logic

multi_step(Next, X, _, _, _, _, _, _, Outcome) :-
    phase(Next, X, Outcome). 