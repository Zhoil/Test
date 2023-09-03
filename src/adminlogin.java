import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
public class adminlogin extends JFrame{
    public adminlogin() {
        setTitle("登录窗口"); 			 					//设置标题
        setLayout(null);     								//设置绝对布局

        Container c = getContentPane(); 					//定义一个容器
        final JTextField jtf1 = new JTextField(); 			//用户名文本框
        final JPasswordField jpf1 = new JPasswordField();	//密码文本框
        JLabel jl1 = new JLabel("用户名:"); 				//“用户名”标签
        JLabel jl2 = new JLabel("密码:");//“密码：”标签
        jpf1.setEchoChar('*');								//设置密码字符为*
        JButton jb1 = new JButton("确定");					//“确定”按钮
        JButton jb2 = new JButton("取消");					//“取消”按钮
        JButton jb3 = new JButton("注册");                  //"注册"按钮
        final int[] fa = {0};

        jb1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                String name = jtf1.getText().trim();
                String psw = new String(jpf1.getPassword());
                if (jtf1.getText().trim().equals("admin") && (new String(jpf1.getPassword())).trim().equals("123") ) {
                    JOptionPane.showMessageDialog(null, "用户名：" + name + "\n密码：" + psw);
                }else if (jtf1.getText().trim().length() == 0 || new String(jpf1.getPassword()).trim().length() == 0) {
                    JOptionPane.showMessageDialog(null, "不能为空!");
                }
                else  {
                    JOptionPane.showMessageDialog(null, "用户名或密码错误");
                    fa[0]++;
                    if(fa[0]>3) {
                        JOptionPane.showMessageDialog(null, "三次以上错误！！！");
                        System.exit(0);
                    }
                }
            }
        });

        jb2.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                int i = JOptionPane.showConfirmDialog(null, "你确定要退出本系统吗？");
                if (i == 0) {
                    System.exit(0);
                }
            }
        });

        jb3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                setVisible(false);
                new Userlogin_to();
            }
        });

        //添加组件到容器
        c.add(jl1);
        c.add(jtf1);
        c.add(jl2);
        c.add(jpf1);
        c.add(jb1);
        c.add(jb2);
        c.add(jb3);
        //设置各组件的位置以及大小
        jl1.setBounds(35, 40, 180, 60);
        jtf1.setBounds(120, 40, 420, 60);
        jl2.setBounds(50, 120, 180, 60);
        jpf1.setBounds(120, 120, 420, 60);
        jb1.setBounds(60, 200, 140, 100);
        jb2.setBounds(230, 200, 140, 100);
        jb3.setBounds(400,200,140,100);
        //设置窗体大小、关闭方式、不可拉伸
        setSize(600, 440);
        setLocationRelativeTo(null);
        setVisible(true);
        setResizable(true);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

    }
    public static void main(String[] args) {

        new adminlogin();
        Toolkit toolkit = Toolkit.getDefaultToolkit();
        Image logo = toolkit.getImage("logo.jpeg");
        JFrame jf = new JFrame();
        jf.setIconImage(logo);

    }
}