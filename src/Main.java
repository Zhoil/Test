import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Random;

class Flight {
    private String flightNumber;
    private String startPoint;
    private String endPoint;
    private String aircraftNumber;
    private int dayOfWeek;
    private int passengerQuota;
    private double ticketPrice;
    Random random = new Random();
    private int remainingTickets;

    public Flight(String flightNumber, String startPoint, String endPoint, String aircraftNumber, int dayOfWeek, int passengerQuota, double ticketPrice) {
        this.flightNumber = flightNumber;
        this.startPoint = startPoint;
        this.endPoint = endPoint;
        this.aircraftNumber = aircraftNumber;
        this.dayOfWeek = dayOfWeek;
        this.passengerQuota = passengerQuota;
        this.ticketPrice = ticketPrice;
        int num = random.nextInt(passengerQuota-10);
        this.remainingTickets = num;
    }

    public int getRemainingTickets() {
        return remainingTickets;
    }

    public void setRemainingTickets(int remainingTickets) {
        this.remainingTickets = remainingTickets;
    }

    public String toString() {
        return flightNumber + " " + startPoint + " " + endPoint;
    }
}

class TicketingSystem {
    private JFrame frame;
    private JTextArea outputArea;
    private JComboBox<Flight> flightComboBox;

    public TicketingSystem() {
        frame = new JFrame("机票订票系统");
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
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

        final boolean[] fla = {false};

        bookButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Flight selectedFlight = (Flight) flightComboBox.getSelectedItem();
                if(!fla[0]) sout(selectedFlight);
                fla[0] = true;
                bookTicket(selectedFlight);
            }
        });
        panel.add(bookButton);

        frame.add(panel, BorderLayout.SOUTH);
    }

    public void addFlight(Flight flight) {
        flightComboBox.addItem(flight);
    }
    public void sout(Flight ff){
        outputArea.append("原剩余票量:" + ff.getRemainingTickets() + "\n\n\n\n");
    }

    public void display() {
        frame.setVisible(true);
    }

    private void bookTicket(Flight flight) {
        if (flight.getRemainingTickets() > 0) {
            flight.setRemainingTickets(flight.getRemainingTickets() - 1);
            outputArea.append("订票成功：" + flight.toString() + "\n" + "现存余票:" + flight.getRemainingTickets() + "\n\n");
        } else {
            outputArea.append("订票失败，该航班已无余票：" + flight.toString() + "\n");
        }
    }
}

public class Main {
    public static void main(String[] args) {
        TicketingSystem ticketingSystem = new TicketingSystem();

        // 添加航班信息
        Flight flight1 = new Flight("F001", "Beijing", "Shanghai", "A001", 1, 100, 500.0);
        Flight flight2 = new Flight("F002", "Shanghai", "Guangzhou", "A002", 2, 200, 800.0);
        ticketingSystem.addFlight(flight1);
        ticketingSystem.addFlight(flight2);

        ticketingSystem.display();
    }
}