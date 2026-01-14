nn(net5, [X], Z, [benign, malicious]) :: phase5(X, Z).

ddos(X, P1, P2, P3, P4, alarm) :-
    4 is P1 + P2 + P3 + P4,
    phase5(X, malicious).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    \+ ddos(X, P1, P2, P3, P4, alarm).