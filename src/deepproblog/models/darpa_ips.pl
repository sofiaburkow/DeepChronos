:- use_module('src/deepproblog/logic/logic.py').

nn(net1, [X], Z, [benign, phase1]) :: ip_sweep(X, Z).
nn(net2, [X], Z, [benign, phase2]) :: sadmind_ping(X, Z).
nn(net3, [X], Z, [benign, phase3]) :: sadmind_explotation(X, Z).
nn(net4, [X], Z, [benign, phase4]) :: ddos_installation(X, Z).
nn(net5, [X], Z, [benign, phase5]) :: ddos_attack(X, Z).

phase(1, X, VictimIP, Outcome) :-
    is_homenet(VictimIP),
    ip_sweep(X, Outcome).

phase(1, X, VictimIP, Outcome) :-
    \+ is_homenet(VictimIP),
    ip_sweep(X, benign),
    Outcome = benign.

phase(2, X, DIP, Outcome) :-
    sadmind_ping(X, Outcome).

phase(3, X, VictimIP, Outcome) :-
    is_homenet(VictimIP),
    sadmind_explotation(X, Outcome).

phase(3, X, VictimIP, Outcome) :-
    \+ is_homenet(VictimIP),
    sadmind_explotation(X, benign),
    Outcome = benign.

phase(4, X, DIP, Outcome) :-
    ddos_installation(X, Outcome).

phase(5, X, VictimIP, Outcome) :-
    \+ is_homenet(VictimIP),
    ddos_attack(X, Outcome).

phase(5, X, VictimIP, Outcome) :-
    is_homenet(VictimIP),
    ddos_attack(X, benign),
    Outcome = benign.

next_phase(P1, P2, P3, P4, Next) :-
    Next is P1 + P2 + P3 + P4 + 1.

multi_step_ips(X, SIP, DIP, P1, P2, P3, P4, Outcome) :-
    next_phase(P1, P2, P3, P4, Next),
    phase(Next, X, DIP, Outcome).