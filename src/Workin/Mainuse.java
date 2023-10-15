package Workin;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Mainuse extends JFrame {
    private JTextArea outputArea;
    private JComboBox<fight> flightComboBox;
    public Mainuse(fight an){
        setTitle("总窗口");
        setLayout(null);
        JPanel jp = new JPanel(); //创建个JPanel
        jp.setOpaque(false); //把JPanel设置为透明 这样就不会遮住后面的背景



        ((JPanel)this.getContentPane()).setOpaque(false);
        ImageIcon img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\980.jpg"); //添加图片
        JLabel background = new  JLabel(img);
        this.getLayeredPane().add(background, new Integer(Integer.MIN_VALUE));
        background.setBounds(0, 0, img.getIconWidth(), img.getIconHeight());


        Container use = getContentPane();



        final JLabel mess = new JLabel("<html><body><font size='7'>欢迎来到航班系统!</font></body></html>");

        final JButton button1 = new JButton("查询航班信息(限单人二等票)");
        final JButton button2 = new JButton("航班——订票");
        final JButton button3 = new JButton("航班——退票");
        final JButton button4 = new JButton("航班——改签");

        final JButton exit = new JButton("退出");


        button1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int i = JOptionPane.showConfirmDialog(null, "选择 '是' 进入单人购票, '否' 进入单机查询.");
                if (i == 0) {
                    dispose();
                    TicketingSystem ticketingSystem = new TicketingSystem();

                    // 添加航班信息
                    fight flight1 = new fight();
                    flight1.flight("F001", "Beijing", "Shanghai", "A001","six");
                    fight flight2 = new fight();
                    flight2.flight("F002", "Shanghai", "Guangzhou", "A002","seven");
                    ticketingSystem.addFlight(flight1);
                    ticketingSystem.addFlight(flight2);

                    ticketingSystem.display();
                }
                else if(i == 1) {
                    dispose();
                    new Flightmessage(an);
                }
            }
        });

        button2.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                new Flightbuy(an);
            }
        });

        button3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                new Flightreturn(an);
            }
        });

        button4.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                fight change = new fight();
                new Flightchange(change);
            }
        });

        exit.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                new Userlogin();
            }
        });

        use.add(jp);
        use.add(mess);
        use.add(button1);
        use.add(button2);
        use.add(button3);
        use.add(button4);
        use.add(exit);



        mess.setBounds(85, 200, 700, 600);
        button1.setHorizontalAlignment(SwingConstants.CENTER);
        button1.setBounds(140,100,200,60);
        button2.setHorizontalAlignment(SwingConstants.CENTER);
        button2.setBounds(140,170,200,60);
        button3.setHorizontalAlignment(SwingConstants.CENTER);
        button3.setBounds(140,240,200,60);
        button4.setHorizontalAlignment(SwingConstants.CENTER);
        button4.setBounds(140,310,200,60);
        exit.setHorizontalAlignment(SwingConstants.CENTER);
        exit.setBounds(140,380,200,60);


        setSize(500, 700);
        setLocationRelativeTo(null);
        setVisible(true);
        setResizable(true);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }
    public void addFlight(fight flight) {
        flightComboBox.addItem(flight);
    }

//    public static void main(String[] args) {
//        fight o = new fight();
//        o.flight("dadaw","eaera","ra2e","rwar","rw213");
//        Mainuse mu = new Mainuse(o);
//        mu.addFlight(o);
//
//    }

}
