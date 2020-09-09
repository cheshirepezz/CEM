%  Script based on leapfrog.m
%  Comments added by Paul Gibbon
close all
clear all


x0=[0,0,0];  % initial coords
v0=[1e3,0,1e2];

E0=[0,10,0];  % initial fields
B0=[0,0,.1];
B1=[1e-4,0,0]*0;  % gradient B perturbation

e=1.602176e-19;
m=9.109e-31;
qom=e/m; % charge/mass ratio

wc=qom*norm(B0); % cyclotron frequency

Time=2*2*pi/wc; % total runtime
dt=.1/wc; % timestep

NT=round(Time/dt);  % # timesteps

v=v0-.5*dt*qom*(E0+cross(v0,B0)); % shift initial velocity back 1/2 step
x=x0;
for it=1:NT
  x=x+dt*v;
  E=E0;
  B=B0;
%    B=B0+B1*x(2);
%
% Boris mover: solves dv/dt = q/m*(E + vxB) to 2nd order accuracy in dt
%
  qomdt2=dt*qom/2;
  h=qomdt2*B;
  s=2*h/(1+dot(h,h));
  u=v+qomdt2*E;
  up=u+cross(u+cross(u,h),s);
  v=up+qomdt2*E;
%
% Bad Mover
%
%v=v+dt*qom*(E+cross(v,B));
%
% TODO: currently plots one point at a time, which slows down the script - need
% to store the orbit in array first, then plot curves after integration complete.
% 
  subplot(2,2,1);    
  color='r';
  x_limit=1.5e-7; % set axis limits
  y_limit=1.5e-7;
  axis([-x_limit, x_limit, -y_limit, y_limit],"square");  % create square box
  plot(x(1),x(2),'o');
  xlabel('x');
  ylabel('y');
  hold on;
  subplot(2,2,2);    
  plot(x(2),v(2),'*');
  xlabel('y');
  ylabel('vy');
  hold on;
  subplot(2,2,3);  
  plot(x(1),v(1),'*');  
  xlabel('x');
  ylabel('vx');
  hold on;
  subplot(2,2,4);    
  plot(x(3),v(3),'*');
  xlabel('z');
  ylabel('vz');
  hold on;
end   
