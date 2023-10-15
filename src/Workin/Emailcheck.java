package Workin;

import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;
import java.util.Date;
import java.util.Properties;
public class Emailcheck {

    public static String myEmailPassword = "evholyxzkqxzecgd";//  *授权码

    public static String myEmailSMTPHost = "smtp.qq.com";

    public static String myEmailAccount="2437537088@qq.com";//   *发件人邮箱，（自己邮箱）
    public static String receiveMailAccount;
    public void send(String check,String account) {    // 外部创建对象，只需调用send方法，第一个参数是验证码，第二个是收件人邮箱（可以是自己的）
        receiveMailAccount = account;
        //1.创建参数配置, 用于连接邮件服务器的参数配置
        Properties props = new Properties();
        props.setProperty("mail.transport.protocol", "smtp");
        props.setProperty("mail.smtp.host", myEmailSMTPHost);
        props.setProperty("mail.smtp.auth", "true");
        Session session = Session.getInstance(props);
        MimeMessage message = null;
        try {
            message = createMimeMessage(check,session, myEmailAccount, receiveMailAccount);
            Transport transport = session.getTransport();

            transport.connect(myEmailAccount, myEmailPassword);
            transport.sendMessage(message, message.getAllRecipients());
            System.out.println("邮件发送成功");
            transport.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    public static MimeMessage createMimeMessage(String x,Session session, String sendMail, String receiveMail) throws Exception {
        // 1.创建一封邮件
        String mss;
        mss=" 欢迎使用航班管理系统，你的验证码为："+x;
        MimeMessage message = new MimeMessage(session);
        // 2.From:发件人（昵称有广告嫌疑，避免被邮件服务器误认为是滥发广告以至返回失败，请修改昵称）
        message.setFrom(new InternetAddress(sendMail, "黑市", "UTF-8"));
        // 3.To:收件人（可以增加多个收件人、抄送、密送）
        message.setRecipient(MimeMessage.RecipientType.TO, new InternetAddress(receiveMail, "XX用户", "UTF-8"));
        // 4.Subject: 邮件主题（标题有广告嫌疑，避免被邮件服务器误认为是滥发广告以至返回失败，请修改标题）
        message.setSubject("航班管理系统验证", "UTF-8");
        // 5.Content: 邮件正文（可以使用html标签）（内容有广告嫌疑，避免被邮件服务器误认为是滥发广告以至返回失败，请修改发送内容）
        message.setContent(mss, "text/html;charset=UTF-8");
        // 6.设置发件时间
        message.setSentDate(new Date());
        // 7.保存设置
        message.saveChanges();
        return message;
    }
}


