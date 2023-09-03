import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.io.IOException;
import java.net.URI;


public class Userlogin_to extends JFrame {
    public Userlogin_to(){
        setTitle("注册窗口");
        setLayout(null);
        setBackground(Color.blue);
        Container a = getContentPane();
        Runtime o = Runtime.getRuntime();

        final JTextField user_1_t = new JTextField("8位数字",10);
        final JPasswordField user_1_p = new JPasswordField();

        JLabel user_1_tl = new JLabel("新建用户名:");
        JLabel user_1_pl = new JLabel("密码(不少于5位):");
        user_1_p.setEchoChar('*');

        JButton jjb1 = new JButton("QQ登陆");
        JButton jjb2 = new JButton("邮箱注册");
        JButton jjb3 = new JButton("返回");

        user_1_t.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent e) {
                if(user_1_t.getText().trim().equals("8位数字")){
                    user_1_t.setText("");
                    user_1_t.setForeground(Color.black);
                }
            }

            @Override
            public void focusLost(FocusEvent e) {
                if(user_1_t.getText().trim().isEmpty()){
                    user_1_t.setForeground(Color.gray);
                    user_1_t.setText("8位数字");
                }
            }
        });


        jjb1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                try {
                    o.exec("D:\\Download\\newQQ\\QQ.exe");
                } catch (IOException ex) {
                    throw new RuntimeException(ex);
                }
            }
        });

        jjb2.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    String urll = "https://mail.163.com/js6/main.jsp?sid=IBrjCC" +
                            "hhxxFHmzwpxshhDwKEZWSDrwus&df=mail163_" +
                            "letter#module=welcome.Welcome" +
                            "Module%7C%7B%7D";
                    //创建一个URI实例
                    URI urii = URI.create(urll);
                    // getDesktop()返回当前浏览器上下文的 Desktop 实例。
                    //Desktop 类允许 Java 应用程序启动已在本机桌面上注册的关联应用程序，以处理 URI 或文件。
                    Desktop dp = Desktop.getDesktop();
                    //判断系统桌面是否支持要执行的功能
                    if (dp.isSupported(Desktop.Action.BROWSE)) {
                        //启动默认浏览器来显示 URI
                        dp.browse(urii);
                    }
                } catch (Exception e1) {
                    e1.printStackTrace();
                }
            }
        });

        jjb3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                new adminlogin();
            }
        });


        a.add(user_1_t);
        a.add(user_1_tl);
        a.add(user_1_p);
        a.add(user_1_pl);
        a.add(jjb1);
        a.add(jjb2);
        a.add(jjb3);


        user_1_tl.setBounds(45, 40, 180, 60);
        user_1_t.setBounds(120, 40, 420, 60);
        user_1_pl.setBounds(20, 120, 180, 60);
        user_1_p.setBounds(120, 120, 420, 60);
        jjb1.setBounds(60, 200, 140, 100);
        jjb2.setBounds(230, 200, 140, 100);
        jjb3.setBounds(400,200,140,100);

        setSize(600, 440);
        setLocationRelativeTo(null);
        setVisible(true);
        setResizable(true);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }
}
