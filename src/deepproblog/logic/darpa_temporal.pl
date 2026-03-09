%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Phase Detectors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nn(net1, [X], Z, [benign, attack]) :: recon_det(X, Z).
nn(net2, [X], Z, [benign, attack]) :: ping_det(X, Z).
nn(net3, [X], Z, [benign, attack]) :: overflow_det(X, Z).
nn(net4, [X], Z, [benign, attack]) :: install_det(X, Z).
nn(net5, [X], Z, [benign, attack]) :: ddos_det(X, Z).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Alert abstractions 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alert(recon,T) :-
    recon_det(T,attack).

alert(ping,T) :-
    ping_det(T,attack).

alert(exploit,T) :-
    exploit_det(T,attack).

alert(install,T) :-
    install_det(T,attack).

alert(ddos,T) :-
    ddos_det(T,attack).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Attack graph edges
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

edge(recon,ping).
edge(ping,exploit).
edge(exploit,install).
edge(install,ddos).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temporal ordering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

before(T1,T2) :-
    T1 < T2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Attack chain reasoning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

chain(Type,T) :-
    alert(Type,T).

chain(Type2,T2) :-
    edge(Type1,Type2),
    chain(Type1,T1),
    alert(Type2,T2),
    before(T1,T2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final attack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

multi_step_attack :-
    chain(ddos,_).