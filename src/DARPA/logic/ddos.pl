nn(net5, [X], Z, [benign, phase5]) :: phase(5, X, Z).

ddos(X, P1, P2, P3, P4, alarm) :-
    4 is P1 + P2 + P3 + P4,
    phase(5, X, phase5).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    \+ ddos(X, P1, P2, P3, P4, alarm).