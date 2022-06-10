f = 0.5; %����Ƶ�ʣ�
w = 2*pi*f; %�����Ƶ�ʣ�
nMax = 400; %�������������
ts = 1/10; %�������ʱ������
n = 0:nMax; %�������У�
x = exp(1i*w*n*ts);%��ָ��������ʽ��

plot3(n*ts, real(x), imag(x));
t = n*ts;
close(mov);
mov = VideoWriter('exp', 'MPEG-4'); %�������ɶ����ļ�������ʽ��
mov.FrameRate = 20; %���嶯�����ŵ�֡�ʣ�
mov.Quality = 20; %������Ƶ������������

open(mov) %����Ƶ�����ļ����вɼ���

for ni = 0:nMax %��ʼѭ����ÿ��ѭ������һ֡ͼ��
    ti = ni*ts; %����ѭ�����ڵĲ���ʱ��㣻
    omega = 1i*w*ti; %��Ƶ�ʣ�
    xi = exp(omega); %��ָ��������
    clf %���֮ǰһ֡��ͼ��������Ϊ�����ɶ�̬ͼ��֮ǰ��ͼ�����������������ָ��ת������Ӧcos��sin��ͼ��
    % phasor
    subplot(2,2,[1,3]); % ����һ��2*2��ͼ��
    hold on %�ڻ�����һ��ͼ��ʱ�������һ��ͼ��
    plot([-1.1,1.1],[0,0],"k"); %����x���߶Σ�
    plot([0,0],[-1.1,1.1],'k'); %����y���߶Σ�
    
    axis equal %data edge equals to axis
    axis([-1.1,1.1,-1.1,1.1]) %set x from -1.1 to 1.1, set y from -1.1 to 1.1
    plot(x); %����x���ף�
    plot(real(xi),imag(xi),'rd'); %���Ƶ�ǰ�����㣻
    plot([0,real(xi)],[0,imag(xi)],'c'); %����ԭ�㵽��������߶Σ�
    xlabel('in-phase');%��ͼ��x���ǩ��
    ylabel('quadrature');%��ͼ��y���ǩ��
    title('phasor');%��ͼ����⣻
    
    %�����������һ����һ�£������ظ���
    
    % cos
    subplot(2,2,2) 
    hold on
    plot([0,4],[0,0],'k');
    axis([0,4,-1.1,1.1]);
    plot(t, real(x));
    plot(ti, real(xi),'rd');
    plot([ti,ti], [0,real(xi)],'c-');
    xlabel('time (s)');
    ylabel('cos(wt)')
    title('cos');
    
    
    % sin
    subplot(2,2,4)
    hold on
    plot([0,4],[0,0],'k');
    axis([0,4,-1.1,1.1]);
    plot(t, imag(x));
    plot(ti, imag(xi),'rd');
    plot([ti,ti],[0,imag(xi)],'c-');
    xlabel('time (s)');
    ylabel('sin(wt)')
    title('sin');
    currentFrame = getframe(gcf);
    writeVideo(mov,currentFrame)

end
close(mov);