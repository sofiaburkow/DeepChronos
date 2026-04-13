nn(net1, [X], Z, [benign, attack]) :: exphil(X, Z).
nn(net2, [X], Z, [benign, attack]) :: recon(X, Z). 
nn(net3, [X], Z, [benign, attack]) :: exploit(X, Z).
nn(net4, [X], Z, [benign, attack]) :: cracking(X, Z).

% Grounding facts

home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

dns(53).
http(80).
https(443).

tcp(6).

internal_traffic(Src, Dst) :-
    home_orig(Src),
    home_resp(Dst).


% Scanning

t(0.7)::horizontal_scan(1).
t(0.9)::vertical_scan(1).
t(0.9)::high_rate_scan(1).

% Phase definitions

multi_step(1, X, Ssr, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, phase1) :-
    internal_traffic(Src, Dst),
    dns(Port),
    exphil(X, attack).

multi_step(1, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, benign) :- 
    \+ multi_step(1, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, phase1).


multi_step(2, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, phase2) :-
    internal_traffic(Src, Dst),
    (horizontal_scan(ScanHo); vertical_scan(ScanVe); high_rate_scan(ScanHi)),
    tcp(Proto),
    recon(X, attack).

multi_step(2, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, benign) :-
    \+ multi_step(2, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, phase2).

    
multi_step(3, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, phase3) :- 
    internal_traffic(Src, Dst),
    https(Port),
    exploit(X, attack).

multi_step(3, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, benign) :-
    \+ multi_step(3, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, phase3).


multi_step(4, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, phase4) :-
    internal_traffic(Src, Dst),
    tcp(Proto),
    (horizontal_scan(ScanHo); vertical_scan(ScanVe); high_rate_scan(ScanHi)),
    cracking(X, attack).

multi_step(4, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, benign) :-
    \+ multi_step(4, X, Src, Dst, Port, Proto, ScanHo, ScanVe, ScanHi, phase4).