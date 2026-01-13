% Neural networks for each phase
% nn(net1, [X], Z, [benign, phase1]) :: phase1(X, Z).
% nn(net2, [X], Z, [benign, phase2]) :: phase2(X, Z).
% nn(net3, [X], Z, [benign, phase3]) :: phase3(X, Z).
% nn(net4, [X], Z, [benign, phase4]) :: phase4(X, Z).
nn(net5, [X], Z, [benign, phase5]) :: phase5(X, Z).

multi_step(X, F1, F2, F3, F4) :-
    phase5(X, Z), 
    F1, F2, F3, F4.

not_multi_step(X, F1, F2, F3, F4) :-
    \+ multi_step(X, _, _, _, _).