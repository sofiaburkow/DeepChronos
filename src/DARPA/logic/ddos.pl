nn(net5, [X], Z, [no_alarm, alarm]) :: phase5(X, Z).

ddos(X, 1, 1, 1, 1, Outcome) :-
    phase5(X, Outcome).
    
ddos(X, P1, P2, P3, P4, no_alarm) :-
    P1 + P2 + P3 + P4 < 4.