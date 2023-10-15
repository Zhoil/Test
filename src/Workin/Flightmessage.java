package Workin;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

//class fight{
//    Random random = new Random();
//    int r1 = random.nextInt(111)+300;
//    int rus1 = random.nextInt(61);
//    int rus2 = random.nextInt(101);
//    int rus3 = random.nextInt(r1-rus1-rus2+1);
//    String tickets1 = Integer.toString(rus1);
//    String tickets2 = Integer.toString(rus2);
//    String tickets3 = Integer.toString(rus3);
//    String flightNumber;
//    String aircraftNumber;
//    String startPoint,endPoint;
//    String dayOfWeek;
//    double[] ticketPrice = {0,894.4,540.5,322.7};
//
//    String passengerQuota = Integer.toString(r1);
//    boolean labe=false;
//
//    public void flight(String flightNumber, String startPoint, String endPoint, String aircraftNumber,
//                       String dayOfWeek) {
//
//        this.flightNumber = flightNumber;
//        this.startPoint = startPoint;
//        this.endPoint = endPoint;
//        this.aircraftNumber = aircraftNumber;
//        this.dayOfWeek = dayOfWeek;
//    }
//    public void leave(int x,int y,int z){
//        this.rus1-=x;
//        this.rus2-=y;
//        this.rus3-=z;
//        labe=true;
//    }
//
//    public void addin(int x,int y,int z){
//            this.rus1+=x;
//            this.rus2+=y;
//            this.rus3+=z;
//    }
//
//}

//class Weather{
//    Message_in t = new Message_in();
//    Document doc = t.getDocument("http://www.weather.com.cn/html/weather/101250101.shtml");
//    // 获取目标HTML代码
//    Elements elements1 = doc.select("[class=sky skyid lv2 on]");
//    // 今天
//    Elements elements2 = elements1.select("h1");
//    String today = elements2.get(0).text();
//    // 天气情况
//    Elements elements3 = elements1.select("[class=wea]");
//    String number = elements3.get(0).text();
//    // 高的温度
//    Elements elements5 = elements1.select("[class=tem]");
//    Elements elements9 = elements5.select("span");
//    String highTemperature = elements9.get(0).text()+"°C";
//    // 低的温度
//    Elements elements10 = elements5.select("i");
//    String lowTemperature = elements10.get(0).text()+"°C";
//    // 风力
//    Elements elements6 = elements1.select("[class=win] i");
//    String wind = elements6.get(0).text();
//
//}

public class Flightmessage extends JFrame {
    private JTextArea outputArea;
    private JComboBox<fight> flightComboBox;
    public Flightmessage(fight mn){     //mn实例传递

        setTitle("信息窗口");
        setLayout(null);
        Container FM = getContentPane();

        JPanel jp = new JPanel(); //创建个JPanel
        jp.setOpaque(false); //把JPanel设置为透明 这样就不会遮住后面的背景



        ((JPanel)this.getContentPane()).setOpaque(false);
        ImageIcon img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\102.jpg"); //添加图片
        JLabel background = new  JLabel(img);
        this.getLayeredPane().add(background, new Integer(Integer.MIN_VALUE));
        background.setBounds(0, 0, img.getIconWidth(), img.getIconHeight());




        final JButton a = new JButton("查询天气");
        final JButton b = new JButton("取消");

        JLabel fN = new JLabel("航班号:"+mn.flightNumber);
        JLabel aN = new JLabel("航次:"+mn.aircraftNumber);
        JLabel d = new JLabel("航周日:"+mn.dayOfWeek);
        JLabel sP = new JLabel("起点:"+mn.startPoint);
        JLabel eP = new JLabel("终点:"+mn.endPoint);
        JLabel ps = new JLabel("票数总额:"+mn.passengerQuota);
        JLabel t1 = new JLabel(mn.ticketPrice[1] + "一等剩余:"+mn.tickets1);
        JLabel t2 = new JLabel(mn.ticketPrice[2] + "二等剩余:"+mn.tickets2);
        JLabel t3 = new JLabel(mn.ticketPrice[3] + "三等剩余:"+mn.tickets3);

        a.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                new Weather(mn);
            }
        });

        b.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(!mn.ll){
                    dispose();
                    new Mainuse(mn);
                }
                else{
                    dispose();
                    TicketingSystem ticketingSystem = new TicketingSystem();

                    // 添加航班信息
                    fight flight1 = new fight();
                    flight1=mn;
                    fight flight2 = new fight();
                    flight2.flight("F002", "Shanghai", "Guangzhou", "A002","seven");
                    ticketingSystem.addFlight(flight1);
                    ticketingSystem.addFlight(flight2);

                    ticketingSystem.display();
                }
            }
        });


        FM.add(a);FM.add(b);FM.add(fN);FM.add(aN);
        FM.add(d);FM.add(sP);FM.add(eP);FM.add(ps);
        FM.add(t1);FM.add(t2);FM.add(t3);


//        fN.setHorizontalAlignment(SwingConstants.CENTER);
        fN.setBounds(180,50,200,40);
        fN.setFont(new Font("华文行楷",Font.BOLD,20));

//        aN.setHorizontalAlignment(SwingConstants.CENTER);
        aN.setBounds(180,100,200,40);
        aN.setFont(new Font("华文行楷",Font.BOLD,20));

//        d.setHorizontalAlignment(SwingConstants.CENTER);
        d.setBounds(180,150,200,40);
        d.setFont(new Font("华文行楷",Font.BOLD,20));

//        sP.setHorizontalAlignment(SwingConstants.CENTER);
        sP.setBounds(180,200,400,40);
        sP.setFont(new Font("华文行楷",Font.BOLD,20));

//        eP.setHorizontalAlignment(SwingConstants.CENTER);
        eP.setBounds(180,250,300,40);
        eP.setFont(new Font("华文行楷",Font.BOLD,20));

//        ps.setHorizontalAlignment(SwingConstants.CENTER);
        ps.setBounds(180,300,300,40);
        ps.setFont(new Font("华文行楷",Font.BOLD,20));

//        t1.setHorizontalAlignment(SwingConstants.CENTER);
        t1.setBounds(180,350,300,40);
        t1.setFont(new Font("华文行楷",Font.BOLD,20));

//        t2.setHorizontalAlignment(SwingConstants.CENTER);
        t2.setBounds(180,400,300,40);
        t2.setFont(new Font("华文行楷",Font.BOLD,20));

//        t3.setHorizontalAlignment(SwingConstants.CENTER);
        t3.setBounds(180,450,300,40);
        t3.setFont(new Font("华文行楷",Font.BOLD,20));

        a.setBounds(100,500,125,100);
        b.setBounds(275,500,125,100);

        setSize(400, 700);
        setLocationRelativeTo(null);
        setVisible(true);
        setResizable(true);
    }


//    public static void main(String[] args) {
//        fight o = new fight();
//        o.flight("dadaw","eaera","ra2e","rwar","rw213");
//        new Flightmessage(o);
//    }

}
