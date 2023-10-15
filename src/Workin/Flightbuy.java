package Workin;


import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;


public class Flightbuy extends JFrame {
    public Flightbuy(fight bn){
        setTitle("订票窗口");
        setLayout(null);

        Container FB = getContentPane();

        JPanel jp = new JPanel(); //创建个JPanel
        jp.setOpaque(false); //把JPanel设置为透明 这样就不会遮住后面的背景



        ((JPanel)this.getContentPane()).setOpaque(false);
        ImageIcon img = new ImageIcon("D:\\DX4\\Java IDEA\\Test IDEA\\untitled\\src\\Workin\\101.jpg"); //添加图片
        JLabel background = new  JLabel(img);
        this.getLayeredPane().add(background, new Integer(Integer.MIN_VALUE));
        background.setBounds(0, 0, img.getIconWidth(), img.getIconHeight());

        final JTextField jtf1 = new JTextField();//
        final JTextField jtf2 = new JTextField();//
        final JTextField jtf3 = new JTextField();//
        final JTextField one = new JTextField();//
        final JTextField two = new JTextField();//
        final JTextField three = new JTextField();//

        final JButton a = new JButton("确定");
        final JButton b = new JButton("取消");

        JLabel jf1 = new JLabel("姓名:");
        JLabel jf2 = new JLabel("订票量:");
        JLabel jf3 = new JLabel("航班号:");
        JLabel _one = new JLabel("舱位等级1:");
        JLabel _two = new JLabel("舱位等级2:");
        JLabel _three = new JLabel("舱位等级3:");


        a.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String t1 = one.getText().trim();
                String t2 = two.getText().trim();
                String t3 = three.getText().trim();

                int rest1 = t1.compareTo(bn.tickets1);
                int rest2 = t2.compareTo(bn.tickets2);
                int rest3 = t3.compareTo(bn.tickets3);
                if(rest1>0 && rest2>0 && rest3>0){
                    JOptionPane.showMessageDialog(null, "抱歉,对应的余票不足");
                }
                else{
                    JOptionPane.showMessageDialog(null, "恭喜您,订票成功");
                    bn.leave(Integer.parseInt(t1),Integer.parseInt(t2),Integer.parseInt(t3));
                }

            }
        });

        b.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                new Mainuse(bn);
            }
        });

        FB.add(jp);

        FB.add(jtf1);FB.add(jtf2);
        FB.add(jtf3);FB.add(one);
        FB.add(two);FB.add(three);
        FB.add(a);FB.add(b);
        FB.add(jf1);FB.add(jf2);
        FB.add(jf3);FB.add(_one);
        FB.add(_two);FB.add(_three);

        jf1.setBounds(80, 40, 180, 60);
        jtf1.setBounds(120, 40, 300, 60);
        jf2.setBounds(70, 110, 180, 60);
        jtf2.setBounds(120, 110, 300, 60);
        jf3.setBounds(70, 180, 180, 60);
        jtf3.setBounds(120, 180, 300, 60);
        _one.setBounds(50, 250, 180, 60);
        one.setBounds(120, 250, 300, 60);
        _two.setBounds(50, 320, 180, 60);
        two.setBounds(120, 320, 300, 60);
        _three.setBounds(50, 390, 180, 60);
        three.setBounds(120, 390, 300, 60);

        a.setBounds(100,500,125,100);
        b.setBounds(275,500,125,100);


        setSize(500, 700);
        setLocationRelativeTo(null);
        setVisible(true);
        setResizable(true);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }


}
