package Workin;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


public class Userlogin extends JFrame {
    public Userlogin() {
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
        JButton jb3 = new JButton("外置QQ");                   //"QQ"程序
        JButton jb4 = new JButton("注册");                 //"注册"按钮
        final int[] fa = {0};

        final int[] ans = {0};

        jb1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                String uPath = "user.txt";
                String pPath = "pass.txt";
                boolean l = false;

                try {
                    // 创建文件读取流
                    FileReader uRead = new FileReader(uPath);
                    FileReader pRead = new FileReader(pPath);
                    BufferedReader uuReader = new BufferedReader(uRead);
                    BufferedReader ppReader = new BufferedReader(pRead);

                    String U;
                    String P;

                    // 读取文件数据
                    while ((U = uuReader.readLine()) != null && (P = ppReader.readLine())!=null) {
                        // 处理读取到的数据
                        if(U.equals(jtf1.getText().trim()) && P.equals(new String(jpf1.getPassword()).trim())){
                            JOptionPane.showMessageDialog(null, "登陆成功");
                            l=true;
                            break;
                        }
                        // 自动换行并继续读取下一行数据
                        else System.out.println(); // 换行
                    }

                    // 关闭读取流
                    uuReader.close();
                    ppReader.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
                ans[0]++;

                if(!l){
                    JOptionPane.showMessageDialog(null, "用户名或密码错误");
                    if(ans[0]>=5){
                        JOptionPane.showMessageDialog(null, "错误5次以上,退出系统!");
                        dispose();
                    }
                    jtf1.setText("");
                    jpf1.setText("");
                }
                else{
                    dispose();
                    fight kl = new fight();
                    new Mainuse(kl);
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
                Runtime o = Runtime.getRuntime();
                try {
                    o.exec("D:\\Download\\newQQ\\QQ.exe");
                } catch (IOException ex) {
                    throw new RuntimeException(ex);
                }
            }
        });

        jb4.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                setVisible(false);
                new Rigister();
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
        c.add(jb4);
        //设置各组件的位置以及大小
        jl1.setBounds(35, 40, 180, 60);
        jtf1.setBounds(120, 40, 420, 60);
        jl2.setBounds(50, 120, 180, 60);
        jpf1.setBounds(120, 120, 420, 60);

        jb1.setBounds(20, 200, 125, 100);
        jb2.setBounds(165, 200, 125, 100);
        jb3.setBounds(310,200,125,100);
        jb4.setBounds(455,200,125,100);
        //设置窗体大小、关闭方式、不可拉伸
        setSize(600, 440);
        setLocationRelativeTo(null);
        setVisible(true);
        setResizable(true);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

    public static void main(String[] args) {
        new Userlogin();
    }
}
