package Workin;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;

public class Rigister extends JFrame {
    public Rigister(){

        setTitle("注册窗口");
        setLayout(null);
        setBackground(Color.blue);

        Container b = getContentPane();
        Runtime o = Runtime.getRuntime();

        final JTextField user_t = new JTextField();
        final JPasswordField user_p = new JPasswordField();

        JLabel user_tl = new JLabel("新建用户名:");
        JLabel user_pl = new JLabel("密码(不少于5位):");
        user_p.setEchoChar('*');
        JButton cd = new JButton("注册");
        JButton sd = new JButton("返回");


         String name = user_t.getText().trim();
         String psw = new String(user_p.getPassword());

         cd.addActionListener(new ActionListener() {
             @Override
             public void actionPerformed(ActionEvent e) {
                 String user = user_t.getText().trim(); // 获取文本框的值
                 String pass = new String(user_p.getPassword()).trim();
                 // 文件路径
                 String userPath = "user.txt";
                 String passPath = "pass.txt";
                 boolean l=false;
                 try {
                     // 创建文件读取流
                     FileReader uRead = new FileReader(userPath);
                     BufferedReader uuReader = new BufferedReader(uRead);

                     String U;

                     // 读取文件数据
                     while ((U = uuReader.readLine()) != null) {
                         // 处理读取到的数据
                         if(U.equals(user)){
                             JOptionPane.showMessageDialog(null, "用户名已被注册过");
                             user_t.setText("");
                             user_p.setText("");
                             l=true;
                             break;
                         }
                         // 自动换行并继续读取下一行数据
                         else System.out.println(); // 换行
                     }

                     // 关闭读取流
                     uuReader.close();
                 } catch (IOException ex) {
                     ex.printStackTrace();
                 }
                 if(!l){
                     try {
                         // 创建文件写入流
                         FileWriter uWriter = new FileWriter(userPath, true);
                         BufferedWriter uuWriter = new BufferedWriter(uWriter);

                         FileWriter pWriter = new FileWriter(passPath, true);
                         BufferedWriter ppWriter = new BufferedWriter(pWriter);
                         // 写入数据到文件
                         uuWriter.write(user);
                         uuWriter.newLine();

                         ppWriter.write(pass);
                         ppWriter.newLine();

                         // 关闭写入流
                         uuWriter.close();
                         ppWriter.close();
                     } catch (IOException ex) {
                         ex.printStackTrace();
                     }

                     JOptionPane.showMessageDialog(null, "注册成功");
                     new Userlogin();
                 }

             }
         });

         sd.addActionListener(new ActionListener() {
             @Override
             public void actionPerformed(ActionEvent e) {
                 dispose();
                 new Userlogin();
             }
         });

          b.add(user_t);
          b.add(user_tl);
          b.add(user_p);
          b.add(user_pl);
          b.add(cd);
          b.add(sd);

          user_tl.setBounds(35, 40, 180, 60);
          user_tl.setForeground(Color.black);
          user_tl.setFont(new Font("华文行楷",Font.BOLD,13));

          user_t.setBounds(120, 40, 420, 60);
          user_pl.setBounds(15, 120, 180, 60);
          user_pl.setForeground(Color.black);
          user_pl.setFont(new Font("华文行楷",Font.BOLD,13));

          user_p.setBounds(120, 120, 420, 60);

          cd.setBounds(50,220,200,100);
          sd.setBounds(350,220,200,100);


          setSize(600, 440);
          setLocationRelativeTo(null);
          setVisible(true);
          setResizable(true);
          setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

}
