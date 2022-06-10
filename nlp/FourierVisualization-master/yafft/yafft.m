t= pi/2:0.1:2.5*pi;

p=1-sin(t);
x=p.*cos(t);
y=p.*sin(t);
ya=x+i*0.4*y;
plot(ya,'b');

plot(t,real(ya),'b');
hold on
plot(t,imag(ya),'r');
hold off
figure;
plot3(t,x,y)

