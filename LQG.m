m = 1
M = 10
g = -10
L = 1



s = 1;
A = [0 1 0 0;
    0 -d/M -m*g/M 0;
    0 0 0 1;
    0 -s*d/(M*L) -s*(m+M)*g/(M*L) 0];
B = [0;1/M; 0 ; s*1/(M*L)];

Q = [1 0 0 0;
    0 1 0 0;
    0 0 10 0;
    0 0 0 100];
R = 0.0001;
K = lqr(A,B,Q,R); % Design controller u = -K*x
%%
C = [1 0 0 0];
D = zeros(size(C,1),size(B,2));

% Augment system with disturbances and noise
vd = .001*eye(4);
vn = .001;

% Build Kalman Filter
[Kf,P,E] =  lqe(A,vd,C,vd,wn);
Kf = (lqr(A',C',vd,vn))';
sysKF = ss(A-Kf*c,[B Kf],eye(4),0*[B Kf]);
