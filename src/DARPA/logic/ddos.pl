nn(net5, [X], Z, [benign, mal]) :: phase5(X, Z).

ddos(X, P1, P2, P3, P4, alarm) :-
    P1 = 1, P2 = 1, P3 = 1, P4 = 1,
    phase5(X, mal).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    phase5(X, benign).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    P1 \= 1.

ddos(X, P1, P2, P3, P4, no_alarm) :-
    P2 \= 1.

ddos(X, P1, P2, P3, P4, no_alarm) :-
    P3 \= 1.

ddos(X, P1, P2, P3, P4, no_alarm) :-
    P4 \= 1.
