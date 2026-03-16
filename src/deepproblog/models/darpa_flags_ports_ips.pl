:- use_module('src/deepproblog/models/logic.py').

% Neural networks

nn(net1, [X], Z, [benign, phase1]) :: recon(X, Z).
nn(net2, [X], Z, [benign, phase2]) :: ping(X, Z).
nn(net3, [X], Z, [benign, phase3]) :: overflow(X, Z).
nn(net4, [X], Z, [benign, phase4]) :: install(X, Z).
nn(net5, [X], Z, [benign, phase5]) :: ddos(X, Z).

% Vulnerability knowledge

sadmind_known_port(111).
sadmind_known_port(Port) :- Port >= 32771.
% sadmind_followup_port(23).

sadmind_port(P) :- sadmind_known_port(P).

download_port(514).
download_port(1022).

% Phase specific rules

phase(1, X, _, _, _, Outcome) :- 
    recon(X, Outcome).


% multi_target(N) :- N >= 3.

% strong explanation boosts probability

phase(2, X, SIP, DIP, DPort, phase2) :-
    external_ip(SIP),
    homenet_ip(DIP),
    sadmind_port(DPort),
    ping(X, phase2).

0.1 :: phase(2, X, _, _, _, Outcome) :-
    ping(X, Outcome).


phase(3, X, _, _, VictimPort, phase3) :- 
    sadmind_port(VictimPort);
    overflow(X, phase3).

phase(3, X, _, _, _, Outcome) :-
    overflow(X, Outcome).


% phase(4, X, _, _, DPort, phase4) :-
%     download_port(DPort); sadmind_followup_port(DPort),
%     install(X, phase4).

phase(4, X, _, _, DPort, phase4) :-
    download_port(DPort);
    install(X, phase4).


phase(4, X, _, _, _, Outcome) :-
    install(X, Outcome).


phase(5, X, _, _, _, Outcome) :-
    ddos(X, Outcome).


% Evidence based confidence

0.20 :: support_level(0).
0.60 :: support_level(1).
0.95 :: support_level(2).


% Multi-step attack reasoning

multi_step(X, Next, Evidence, SIP, DIP, DPort, Outcome) :-
    phase(Next, X, SIP, DIP, DPort, Outcome),
    support_level(Evidence).