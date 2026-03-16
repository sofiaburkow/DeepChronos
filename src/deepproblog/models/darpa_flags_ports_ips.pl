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
sadmind_port(P) :- sadmind_known_port(P).

download_port(514).
download_port(1022).

icmp(1).
tcp(6).
udp(17).

loc_orig(1).
loc_resp(1).
ext_orig(0).
ext_resp(0).

% Multi-step attack logic

phase(1, X, SO, DO, _, Proto, phase1) :-
    ext_orig(SO),
    loc_resp(DO),
    icmp(Proto).

phase(1, X, _, _, _, _, Pred) :- 
    recon(X, Pred).


phase(2, _, SO, DO, DPort, Proto, phase2) :-
    ext_orig(SO),
    loc_resp(DO),
    sadmind_port(DPort),
    udp(Proto).

phase(2, _, SO, DO, DPort, Proto, phase2) :-
    loc_resp(SO),
    ext_orig(DO),
    sadmind_port(DPort),
    icmp(Proto).

phase(2, X, _, _, _, _, Pred) :-
    ping(X, Pred).


phase(3, X, _, _, VictimPort, _, phase3) :- 
    sadmind_port(VictimPort).

phase(3, X, _, _, _, _, Pred) :-
    overflow(X, Pred).


phase(4, X, _, _, DPort, _, phase4) :-
    download_port(DPort).

phase(4, X, _, _, _, _, Pred) :-
    install(X, Pred).


phase(5, X, _, _, _, _, Pred) :-
    ddos(X, Pred).


% 0.20 :: support_level(0).
% 0.60 :: support_level(1).
% 0.95 :: support_level(2).


multi_step(X, Next, _, LocalOrig, LocalResp, DPort, Proto, Outcome) :-
    phase(Next, X, LocalOrig, LocalResp, DPort, Proto, Outcome).
    % support_level(Evidence).