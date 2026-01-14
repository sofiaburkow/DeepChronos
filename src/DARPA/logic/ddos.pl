% Works with pretrained net5
nn(net5, [X], Z, [benign, malicious]) :: phase5(X, Z).

phases_complete(P1, P2, P3, P4) :-
    4 is P1 + P2 + P3 + P4.

ddos(X, P1, P2, P3, P4, alarm) :-
    phases_complete(P1, P2, P3, P4),
    phase5(X, malicious).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    phases_complete(P1, P2, P3, P4),
    phase5(X, benign).

ddos(X, P1, P2, P3, P4, no_alarm) :-
    P1 + P2 + P3 + P4 < 4.