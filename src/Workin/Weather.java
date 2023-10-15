package Workin;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Weather extends JFrame {
    public Weather(fight wn) {
        String city="";
        String temp="";
        String wea="";
        String mes="";
        String week="";

        String ans="";

        try{
            URL url = new URL("http://t.weather.itboy.net/api/weather/city/101250101");
            InputStreamReader isReader =  new InputStreamReader(url.openStream(),"UTF-8");//“UTF- 8”万国码，可以显示中文，这是为了防止乱码
            BufferedReader br = new BufferedReader(isReader);//采用缓冲式读入
            String str;
            while((str = br.readLine()) != null){
                String regex="\\p{Punct}+";
                String digit[]=str.split(regex);
//                System.out.println('\n'+"城市:"+digit[22]+digit[18]);
                city=digit[22]+digit[18];
//                System.out.println('\n'+"时间:"+digit[49]+"年"+digit[50]+"月"+digit[51]+"日"+digit[53]);
                week=digit[53];
//                System.out.println('\n'+"温度:"+digit[47]+"~"+digit[45]);
                temp=digit[47]+"~"+digit[45];
//                System.out.println('\n'+"天气:"+digit[67]+" "+digit[63]+digit[65]);
                wea=digit[67]+" "+digit[63]+digit[65];
                ans=digit[67];
//                System.out.println('\n'+digit[69]);
                mes=digit[69];
            }
            br.close();//网上资源使用结束后，数据流及时关闭
            isReader.close();
        }
        catch(Exception exp){
            System.out.println(exp);
        }

        setTitle("天气");
        setLayout(null);
        Container WT = getContentPane();

        JPanel jp = new JPanel(); //创建个JPanel
        jp.setOpaque(false); //把JPanel设置为透明 这样就不会遮住后面的背景



        ((JPanel)this.getContentPane()).setOpaque(false);
        ImageIcon img = null;  //  设置天气背景的实时转换
        if(ans.contains("晴"))  img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\006.jpg"); //添加图片
        else if (ans.contains("雨")) img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\002.jpg"); //添加图片
        else if (ans.contains("阴")) img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\001.jpg"); //添加图片
        else if (ans.contains("雪")) img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\005.jpg"); //添加图片
        else if (ans.contains("雷")) img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\004.jpg"); //添加图片
        else if (ans.contains("雾")) img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\003.jpg"); //添加图片
        else img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\111.jpg");
        JLabel background = new  JLabel(img);
        this.getLayeredPane().add(background, new Integer(Integer.MIN_VALUE));
        background.setBounds(0, 0, img.getIconWidth(), img.getIconHeight());

        JLabel cit = new JLabel(city);
        JLabel tem = new JLabel(temp);
        JLabel Wea = new JLabel(wea);
        JLabel Mes = new JLabel(mes);

        final JButton a = new JButton("更新");
        final JButton b = new JButton("返回");

        a.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                new Weather(wn);
            }
        });

        b.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                new Flightmessage(wn);
            }
        });


        SimpleDateFormat formatter= new SimpleDateFormat("yyyy-MM-dd  'at'  HH:mm:ss  z");
        Date date = new Date(System.currentTimeMillis());

        JLabel time = new JLabel(formatter.format(date)+" "+week);

        WT.add(jp);
        WT.add(cit);
        WT.add(tem);
        WT.add(Wea);
        WT.add(Mes);
        WT.add(time);

        WT.add(a);WT.add(b);


        cit.setHorizontalAlignment(SwingConstants.CENTER);
        cit.setBounds(140,70,200,50);
        cit.setFont(new Font("华文行楷",Font.BOLD,30));

        tem.setHorizontalAlignment(SwingConstants.CENTER);
        tem.setBounds(140,130,200,50);
        tem.setFont(new Font("华文行楷",Font.BOLD,20));

        Wea.setHorizontalAlignment(SwingConstants.CENTER);
        Wea.setBounds(140,190,200,50);
        Wea.setFont(new Font("华文行楷",Font.BOLD,20));
        Wea.setForeground(Color.orange);

        time.setHorizontalAlignment(SwingConstants.CENTER);
        time.setBounds(60,250,400,50);
        time.setFont(new Font("华文行楷",Font.BOLD,20));

        Mes.setHorizontalAlignment(SwingConstants.CENTER);
        Mes.setBounds(100,310,300,50);
        Mes.setFont(new Font("华文行楷",Font.BOLD,20));
        Mes.setForeground(Color.magenta);


        a.setBounds(100,500,125,100);
        b.setBounds(275,500,125,100);


        setSize(500, 700);
        setLocationRelativeTo(null);
        setVisible(true);
        setResizable(true);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

}
