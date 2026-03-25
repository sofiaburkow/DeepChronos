% Neural networks

nn(net1, [X], Z, [benign, attack]) :: darpa(X, Z).

% Per phase rules

phase(1, X, phase1) :-
    darpa(X, attack).

phase(1, X, benign) :- 
    \+ phase(1, X, phase1).

phase(2, X, phase2) :-
    darpa(X, attack).
    
phase(2, X, benign) :-
    \+ phase(2, X, phase2).
    
phase(3, X, phase3) :- 
    darpa(X, attack).

phase(3, X, benign) :-
    \+ phase(3, X, phase3).

phase(4, X, phase4) :-
    darpa(X, attack).

phase(4, X, benign) :-
    \+ phase(4, X, phase4).

phase(5, X, phase5) :-
    darpa(X, attack).

phase(5, X, benign) :-
    \+ phase(5, X, phase5).


% Multi-step attack logic

multi_step(Next, X, _, _, _, _, _, _, Outcome) :-
    phase(Next, X, Outcome). 