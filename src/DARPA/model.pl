nn(net1, [X], Z, [benign,attack]):: phase1(X,Z).
nn(net2, [X], Z, [benign,attack]):: phase2(X,Z).
% nn(net3, [X], Z, [benign,attack]):: phase3(X,Z).
% nn(net4, [X], Z, [benign,attack]):: phase4(X,Z).
% nn(net5, [X], Z, [benign,attack]):: phase5(X,Z).

recon(attack, benign, alarm).
recon(benign, attack, alarm).
recon(attack, attack, alarm).
recon(benign, benign, no_alarm).

multi_step(X, Y) :-
    phase1(X, Z1),
    phase2(X, Z2),
    recon(Z1, Z2, Y).