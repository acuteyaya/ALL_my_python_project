fs=128;
N=1280;%采集点数
n=0:N-1;
t=n/fs;
y= 2*sin(2*pi*1*t)+sin(2*pi*20*t)+2*sin(2*pi*40*t)+3*sin(2*pi*30*t);
x=fft(y,N);
m=abs(x)/N;
k=m(1:N/2);
n1=0:(N-1)/2;
f=n1*fs/N;  
subplot(1,2,1),plot(f,k);
xlabel('频率/Hz');
ylabel('振幅');title('N=128');


%t=0:N-1
%t=t/fs
yak=ones(1,fs/2);
y= 2*sin(2*pi*1*t)+sin(2*pi*20*t)+2*sin(2*pi*40*t)+3*sin(2*pi*30*t);
for k = 0:fs-1
    ft = exp(-1i*2*pi*k*t);
    ft = ft.*y;
    sum=0;
    for j = 1:N
        sum=sum+ft(j);
        %sum=sum+subs(ft,t,j);
    end
    sum=abs(sum)/N;
    yak(k+1)=sum;
    if(k==fs/2)
        break
    end
end
t=0:fs/2;
subplot(1,2,2),plot(t,yak);