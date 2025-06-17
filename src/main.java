import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class main {
    public static void main(String[] args) {

        ExecutorService pool = Executors.newFixedThreadPool(10);
        pool.submit(() -> System.out.println("任务执行"));

    }

}
