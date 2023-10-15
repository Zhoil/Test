package Workin;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Random;

class fight {
    Random random = new Random();
    int r1 = random.nextInt(111)+300;
    int rus1 = random.nextInt(61);
    int rus2 = random.nextInt(101);
    int rus3 = random.nextInt(r1-rus1-rus2+1);
    int SUM = rus1+rus2+rus3;
    String tickets1 = Integer.toString(rus1);
    String tickets2 = Integer.toString(rus2);
    String tickets3 = Integer.toString(rus3);
    String flightNumber;
    String aircraftNumber;
    String startPoint,endPoint;
    String dayOfWeek;
    double[] ticketPrice = {0,894.4,540.5,322.7};

    String passengerQuota = Integer.toString(r1);
    boolean labe=false;
    boolean ll = false;

    public void flight(String flightNumber, String startPoint, String endPoint, String aircraftNumber,
                       String dayOfWeek) {

        this.flightNumber = flightNumber;
        this.startPoint = startPoint;
        this.endPoint = endPoint;
        this.aircraftNumber = aircraftNumber;
        this.dayOfWeek = dayOfWeek;
    }
    public void leave(int x,int y,int z){
        this.rus1-=x;
        this.tickets1 = Integer.toString(this.rus1);
        this.rus2-=y;
        this.tickets2 = Integer.toString(this.rus2);
        this.rus3-=z;
        this.tickets3 = Integer.toString(this.rus3);
        labe=true;
    }

    public void addin(int x,int y,int z){
        this.rus1+=x;
        this.tickets1 = Integer.toString(this.rus1);
        this.rus2+=y;
        this.tickets2 = Integer.toString(this.rus2);
        this.rus3+=z;
        this.tickets3 = Integer.toString(this.rus3);
    }

    public  int getRus2(){
        return rus2;
    }
    public int getRemainingTickets() {
        return r1;
    }

    public void setRemainingTickets(int remainingTickets) {
        this.r1 = remainingTickets;
    }

    public String toString() {
        return flightNumber + " " + startPoint + " " + endPoint;
    }
}

class TicketingSystem {
    private JFrame frame;
    private JTextArea outputArea;
    private JComboBox<fight> flightComboBox;

    public TicketingSystem() {
        frame = new JFrame("机票订票系统");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setSize(500, 400);
        frame.setLayout(new BorderLayout());

        outputArea = new JTextArea();
        outputArea.setEditable(false);
        frame.add(new JScrollPane(outputArea), BorderLayout.CENTER);

        JPanel panel = new JPanel();
        panel.setLayout(new FlowLayout());

        JLabel flightLabel = new JLabel("选择航班：");
        panel.add(flightLabel);

        flightComboBox = new JComboBox<>();
        panel.add(flightComboBox);

        JButton bookButton = new JButton("订票");
        JButton ret = new JButton("返回");
        JButton sear = new JButton("查询");

        bookButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                fight selectedFlight = (fight) flightComboBox.getSelectedItem();
                bookTicket(selectedFlight);
            }
        });

        ret.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.dispose();
                fight selectedFlight = (fight) flightComboBox.getSelectedItem();
                new Mainuse(selectedFlight);
            }
        });

        sear.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.dispose();
                fight selectedFlight = (fight) flightComboBox.getSelectedItem();
                selectedFlight.ll=true;
                new Flightmessage(selectedFlight);
            }
        });
        panel.add(bookButton);
        panel.add(ret);
        panel.add(sear);

        frame.add(panel, BorderLayout.SOUTH);
    }

    public void addFlight(fight flight) {
        flightComboBox.addItem(flight);
    }

    public void display() {
        frame.setVisible(true);
    }

    private void bookTicket(fight flight) {
        if (flight.getRus2() > 0) {
            flight.setRemainingTickets(flight.getRemainingTickets() - 1);
            flight.leave(0,1,0);
//            flight.rus2-=1;
//            flight.r1-=1;
//            flight.tickets2 = Integer.toString(flight.rus2);
//            flight.passengerQuota = Integer.toString(flight.r1);
            outputArea.append("订票成功：" + flight.toString() + "\n" + "剩余 " + flight.tickets2 + "\n\n");
        } else {
            outputArea.append("订票失败，该航班二等票已无余票：" + flight.toString() + "\n");
        }
    }
}

//public class FLI {
//    public static void main(String[] args) {
//        TicketingSystem ticketingSystem = new TicketingSystem();
//
//        // 添加航班信息
//        fight flight1 = new fight();
//        flight1.flight("F001", "Beijing", "Shanghai", "A001","six");
//        fight flight2 = new fight();
//        flight2.flight("F002", "Shanghai", "Guangzhou", "A002","seven");
//        ticketingSystem.addFlight(flight1);
//        ticketingSystem.addFlight(flight2);
//
//        ticketingSystem.display();
//    }
//}