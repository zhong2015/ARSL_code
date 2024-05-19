package recommender.common;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.StringTokenizer;

public class CommonRec {

    public static final int RMSE = 1;//均方根误差
    public static final int MAE = 2;//平均绝对误差
    public static final int R_Squared = 3;

    public static String dataSetName;
    public static String rsDataSetName;
    public static ArrayList<RTuple> trainDataSet =  null;
    public static ArrayList<RTuple> testDataSet =  null;
    public static int trainDataSetSize;

    public static int maxID = 0; //maxID=行数（user的个数）=列数（item的个数）【对称矩阵】

    public static double RSetSize[], CSetSize[];

    public static double lambda = 0.005, lambda1 = 0.005, lambda2 = 0.005; // 正则化参数
    public static double eta = 0.005;
    public static double eta1 = 0.005;
    public static double eta2 = 0.005;
    public static double theta = 0.005; // 增广项参数
    public static double rho = 0.005;
    public static int maxRound = 500; // 最多训练轮数
    public static int featureDimension = 0; // 特征维数
    public static double minGap = 0;
    public static double maxGap = 0;

    public double  max_Res = -100;
    public double maxTotalTime = 0;
    public int max_Round = 0;

    public double  min_Error = 1e10; // 最小误差值
    public double cacheTotalTime = 0;
    public double minTotalTime = 0;
    public int min_Round = 0; // 记录达到最优结果时的最小迭代次数
    public int total_Round = 0;
    public double total_Time = 0;

    public static double[][] catchedP, catchedQ, catchedZ;
    public static int mappingScale = 1000;
    public static double featureInitMax = 0.004;
    public static double featureInitScale = 0.004;

    // 特征矩阵,对应ASNLF的A矩阵
    public double[][] Z, X, Y;

    // 存储特征值的路径
    public static String featureSaveDirP, featureSaveDirQ;

    // ANLF所需的额外参数
    // 对应A，P和对应拉格朗日乘子
    // 进行更新时的缓存矩阵
    public static double[] P_U, P_D, Q_U, Q_D, PQ, QP, XU, YU, XD, YD, XC, YC;
    public static int[] U;
    public static HashSet<Integer> knownNodes =  new HashSet<Integer>();

    //Parallel Hyper-parameters
    public static int trainMaxID = 0;


    public CommonRec() {
        this.initInstanceFeatures();
    }

    /*
     * 初始化实例特征矩阵
     */
    public void initInstanceFeatures() {
        // 加1是为了在序号上与ID保持一致
        X = new double[maxID + 1][featureDimension];
        Y = new double[maxID + 1][featureDimension];
        Z = new double[maxID + 1][maxID + 1];
        for (int i = 1; i <= maxID; i++) {
            for (int j = 0; j < featureDimension; j++) {
                X[i][j] = catchedP[i][j];
                Y[i][j] = catchedQ[i][j];

            }
            for(int k = 1; k <= maxID; k++){
                Z[i][k] = catchedZ[i][k];
            }
        }
    }

    /*
     * 生成初始的训练集、测试集以及统计各个结点的评分数目
     */
    public static void dataLoad(String trainFileName, String testFileName, String separator) throws IOException {
        //生成初始的训练集
        trainDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, trainFileName, trainDataSet,1);

        //生成初始的验证集or测试集
        testDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, testFileName, testDataSet,2);
        initRatingSetSize();
    }

    /*
     * 数据集生成器
     */
    public static void dataSetGenerator(String separator, String fileName, ArrayList<RTuple> dataSet, int flag) throws IOException {

        File fileSource = new File(fileName);
        BufferedReader in = new BufferedReader(new FileReader(fileSource));

        String line;
        while (((line = in.readLine()) != null)){
            StringTokenizer st = new StringTokenizer(line, separator);
            String personID = null;
            if (st.hasMoreTokens())
                personID = st.nextToken();
            String movieID = null;
            if (st.hasMoreTokens())
                movieID = st.nextToken();
            String personRating = null;
            if (st.hasMoreTokens())
                personRating = st.nextToken();
            int iUserID = Integer.valueOf(personID);
            int iItemID = Integer.valueOf(movieID);
            knownNodes.add(iUserID);
            knownNodes.add(iItemID);
            // 记录下最大的itemid和userid；因为itemid和userid是连续的，所以最大的itemid和userid也代表了各自的数目
            maxID = (maxID > iUserID) ? maxID : iUserID;
            maxID = (maxID > iItemID) ? maxID : iItemID;
            double dRating = Double.valueOf(personRating);

            RTuple temp1 = new RTuple();

            temp1.userID = iUserID;
            temp1.itemID = iItemID;
            temp1.rating = dRating;
            dataSet.add(temp1);

            if(iUserID != iItemID){
//                // 不是对角线元素的话，则读入对称项
                RTuple temp2 = new RTuple();
                temp2.userID = iItemID;
                temp2.itemID = iUserID;
                temp2.rating = dRating;
                dataSet.add(temp2);
            }

            // Parallel
            if(flag == 1){
                trainMaxID = (trainMaxID > iUserID) ? trainMaxID : iUserID;
                trainMaxID = (trainMaxID > iItemID) ? trainMaxID : iItemID;
            }
        }
        in.close();
    }


    /*
     * 统计各个结点的评分数目
     */
    public static void initRatingSetSize() {
        RSetSize = new double[maxID + 1];
        CSetSize = new double[maxID + 1];
        for (int i = 0; i <= maxID; i++){
            RSetSize[i] = 0;
            CSetSize[i] = 0;
        }
        for (RTuple tempRating : trainDataSet) {
            RSetSize[tempRating.userID] += 1;
        }
    }

    /*
     * 声明辅助矩阵，并用随机数进行初始化
     */
    public static void initStaticFeatures() throws IOException {

        // 加1是为了在序号上与ID保持一致
        catchedP = new double[maxID + 1][featureDimension];
        catchedQ = new double[maxID + 1][featureDimension];
        catchedZ = new double[maxID + 1][maxID + 1];
        featureSaveDirP = featureSaveDirP + featureDimension + ".txt";
        featureSaveDirQ = featureSaveDirQ + featureDimension + ".txt";
        File featureFileP = new File(featureSaveDirP);        // new File(".") 表示用当前路径 生成一个File实例!!!并不是表达创建一个 . 文件
        File featureFileQ = new File(featureSaveDirQ);
        if(featureFileP.exists()) {
            System.out.println("准备读取指定初始值...");
            readFeatures(catchedP, featureSaveDirP);  // 读取特征矩阵
            System.out.println("读取完毕！！！");
        }else{
            System.out.println("准备生成随机初始值...");
            // 初始化特征矩阵,采用随机值,从而形成一个K阶逼近
            Random random = new Random(System.currentTimeMillis());
            for (int i = 1; i <= maxID; i++) {

                // 特征矩阵的初始值在(0,0.004]
                for (int j = 0; j < featureDimension; j++) {
                    int temp = random.nextInt(mappingScale); //返回[0,mappingScale)随机整数
                    catchedP[i][j] = featureInitMax - featureInitScale * temp / mappingScale;
                }
            }

            // 写入文件
            writeFeatures(catchedP,featureSaveDirP);
            System.out.println("写入P随机初始值完毕！！！");
        }

        if(featureFileQ.exists()) {
            System.out.println("准备读取指定初始值...");
            readFeatures(catchedQ, featureSaveDirQ);  // 读取特征矩阵
            System.out.println("读取完毕！！！");
        }else{
            System.out.println("准备生成随机初始值...");
            // 初始化特征矩阵,采用随机值,从而形成一个K阶逼近
            Random random = new Random(System.currentTimeMillis());
            for (int i = 1; i <= maxID; i++) {

                // 特征矩阵的初始值在(0,0.004]
                for (int j = 0; j < featureDimension; j++) {
                    int temp = random.nextInt(mappingScale); //返回[0,mappingScale)随机整数
                    catchedQ[i][j] = featureInitMax - featureInitScale * temp / mappingScale;
                }
            }

            // 写入文件
            writeFeatures(catchedQ,featureSaveDirQ);
            System.out.println("写入Q随机初始值完毕！！！");
        }

        System.out.println("准备生成随机初始值...");
        // 初始化特征矩阵,采用随机值,从而形成一个K阶逼近
        for (int i = 1; i <= maxID; i++) {
            // 特征矩阵的初始值在(0,0.004]
            for (int j = 1; j <= maxID; j++) {
                catchedZ[i][j] = dotMultiply(catchedP[i], catchedQ[j]);
            }
        }
        System.out.println("生成Z随机初始值完毕！！！");



        // 声明辅助矩阵
        initAuxArray();
    }

    private static void writeFeatures(double[][] catched, String featureSaveDir) throws IOException {

        FileWriter fw = new FileWriter(featureSaveDir);

        for(int i = 1; i <= maxID; i++) {
            for(int k = 0; k < featureDimension; k++) {
                fw.write(catched[i][k] + "::");
            }
            fw.write("\n");
        }
        fw.flush();
        fw.close();
    }

    private static void readFeatures(double[][] catched, String featureSaveDir) throws IOException {

        BufferedReader in = new BufferedReader(new FileReader(featureSaveDir));
        String line;  // 一行数据
        int i = 1;    // 行标
        while((line = in.readLine()) != null){
            String [] temp = line.split("::"); // 各数字之间用"::"间隔
            for(int k = 0; k < featureDimension; k++) {
                catched[i][k] = Double.valueOf(temp[k]);
            }
            i++;
        }
        in.close();
    }

    /*
     * 声明辅助矩阵
     */
    public static void initAuxArray() {

        // 加1是为了在序号上与ID保持一致
        XU = new double[maxID + 1];
        XD = new double[maxID + 1];
        YU = new double[maxID + 1];
        YD = new double[maxID + 1];
        XC = new double[maxID + 1];
        YC = new double[maxID + 1];
    }

    /*
     * 将辅助矩阵的元素置为0(SNLF)
     */
    public static void resetAuxArray() {
        for (int i = 1; i <= maxID; i++) {
            XU[i] = 0;
            XD[i] = 0;
            YU[i] = 0;
            YD[i] = 0;
        }
    }

    /*
     * 计算考虑线性偏差的预测值
     */
    public double getPrediction(int firstID, int secondID) {
        double ratingHat = 0;
        ratingHat += dotMultiply(X[firstID], Y[secondID]);
        return ratingHat;
    }

    // 计算两个向量点乘
    public static double dotMultiply(double[] x, double[] y) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    public double testRMSE() {
        // 计算在测试集上的RMSE
        double sumRMSE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumRMSE += Math.pow((actualRating - ratinghat), 2);
            sumCount++;
        }
        double RMSE = Math.sqrt(sumRMSE / sumCount);
        return RMSE;
    }

    public double testMAE() {
        // 计算在测试集上的MAE
        double sumMAE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumMAE += Math.abs(actualRating - ratinghat);
            sumCount++;
        }
        double MAE = sumMAE / sumCount;
        return MAE;
    }

    public double avg_testData(){
        double sum = 0, count = 0;
        for (RTuple testR : testDataSet) {
            sum += testR.rating;
            count++;
        }
        double result = sum / count;
        return result;
    }

    public double testR_Square(double avg){

        double SSR = 0, SST = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            SSR += Math.pow((actualRating - ratinghat), 2);
            SST += Math.pow((actualRating - avg), 2);
        }
        double result = 1 - SSR / SST;
        return result;
    }

    public void outPutModelSym() throws IOException {

        FileWriter fwUp = new FileWriter(
                new File("./" + rsDataSetName  + "_ModelSym_Up.txt"), true);
        FileWriter fwDown = new FileWriter(
                new File("./" + rsDataSetName  + "_ModelSym_Down.txt"), true);
        fwUp.write("i-j:\n");
        fwDown.write("j-i:\n");
        for(int i = 1; i <= 1500; i++){
            for(int j = 1500; j >= i; j--){
                double ratinghatUp = getPrediction(i, j);
                double ratinghatDown = getPrediction(j, i);
                fwUp.write(ratinghatUp + "\n");
                fwDown.write(ratinghatDown + "\n");
                fwUp.flush();
                fwDown.flush();
            }
        }
        fwUp.close();
        fwDown.close();
    }

}
