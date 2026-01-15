% Original version

nn(net5, [X], Z, [no_alarm, alarm]) :: phase5(X, Z).

ddos(X, P1, P2, P3, P4, Outcome) :-
    4 is P1 + P2 + P3 + P4,
    phase5(X, Outcome).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    P1 + P2 + P3 + P4 < 4.


% Alternative simplified version

nn(net5, [X], Z, [no_alarm, alarm]) :: phase5(X, Z).

ddos(X, 1, 1, 1, 1, Outcome) :-
    phase5(X, Outcome).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    P1 + P2 + P3 + P4 < 4.


% With negation

nn(net5, [X], Z, [benign, malicious]) :: phase5(X, Z).

ddos(X, P1, P2, P3, P4, alarm) :-
    4 is P1 + P2 + P3 + P4,
    phase5(X, malicious).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    \+ ddos(X, P1, P2, P3, P4, alarm). % only symbolic reasoning

