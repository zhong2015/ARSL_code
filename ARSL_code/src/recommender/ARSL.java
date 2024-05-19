package recommender;

import recommender.common.CommonRec;
import recommender.common.RTuple;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class ARSL extends CommonRec {

    public ARSL() {
        super();
    }

    public static void main(String[] args) throws IOException {
        int[] rsArr = new int[]{1};
        for (int rs : rsArr) {
            CommonRec.dataSetName = "zyr_crystm02";
            CommonRec.rsDataSetName = String.valueOf(rs) + "_" + CommonRec.dataSetName;
            String filePath = "D:\\codes\\Test\\DS\\";
            CommonRec.dataLoad(filePath + CommonRec.rsDataSetName + "_train.txt", filePath + CommonRec.rsDataSetName + "_test.txt", "::");
            System.out.println("当前物种的蛋白质总数：" + maxID);
            CommonRec.trainDataSetSize = trainDataSet.size();
            System.out.println("训练集的容量：" + CommonRec.trainDataSetSize);
            System.out.println("测试集的容量：" + CommonRec.testDataSet.size());

            // 设置公共参数
            CommonRec.maxRound = 1000;
            CommonRec.minGap = 1e-5;
            CommonRec.maxGap = 1000;

            for (int tempdim = 20; tempdim <= 20; tempdim += CommonRec.featureDimension) {
                CommonRec.featureDimension = tempdim;
                CommonRec.featureSaveDirP = "./savedLFs/" + CommonRec.dataSetName + "/P";
                CommonRec.featureSaveDirQ = "./savedLFs/" + CommonRec.dataSetName + "/Q";
                // 初始化特征矩阵
                CommonRec.initStaticFeatures();

                experimenter(CommonRec.RMSE);
//                experimenter(CommonRec.MAE);
            }
        }
    }

    /*
     * 综合实验
     */
    public static void experimenter(int metrics) throws IOException {
        long file_tMills = System.currentTimeMillis(); //用于给train函数打开在当前函数所创建的文件
        FileWriter fw;
        if (metrics == CommonRec.RMSE)
            fw = new FileWriter(new File("./" + rsDataSetName + "_RMSE_" + Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills + "dim=" + featureDimension + ".txt"), true);
        else
            fw = new FileWriter(new File("./" + rsDataSetName + "_MAE_" + Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills + "dim=" + featureDimension + ".txt"), true);

        String blankStr = "                          ";
        String starStr = "****************************************************************";
        String equalStr = "=====================================";

        // 按正则项系数lambda的不同取值开始测试
        for (double tempLam1 = Math.pow(2, -18); tempLam1 >= Math.pow(2, -18); tempLam1 *= Math.pow(2, -1)) {

            CommonRec.lambda = tempLam1;

            // 打印标题项
            System.out.println("\n" + starStr);
            System.out.println(blankStr + "featureDimension——>" + CommonRec.featureDimension);
            System.out.println(blankStr + "lambda1——>" + CommonRec.lambda);
            System.out.println(blankStr + "minGap——>" + CommonRec.minGap);
            System.out.println(starStr);

            fw.write("\n" + starStr + "\n");
            fw.write(blankStr + "featureDimension——>" + CommonRec.featureDimension + "\n");
            fw.write(blankStr + "lambda1——>" + CommonRec.lambda + "\n");
            fw.write(blankStr + "minGap——>" + CommonRec.minGap + "\n");
            fw.write(starStr + "\n");
            fw.flush();

            for (double tempLam2 = Math.pow(2, -4); tempLam2 >= Math.pow(2, -4); tempLam2 *= Math.pow(2, -1)) {

                CommonRec.lambda1 = tempLam2;

                // 打印标题项
                System.out.println("\n" + equalStr);
                System.out.println("        lambda2——>" + CommonRec.lambda1);
                System.out.println(equalStr);

                fw.write("\n" + equalStr + "\n");
                fw.write("        lambda2——>" + CommonRec.lambda1 + "\n");
                fw.write(equalStr + "\n");
                fw.flush();

                for (double tempLam3 = Math.pow(2, 6); tempLam3 >= Math.pow(2, 6); tempLam3 *= Math.pow(2, -1)) {
                    CommonRec.lambda2 = tempLam3;

                    // 打印标题项
                    System.out.println("\n" + equalStr);
                    System.out.println("        lambda3——>" + CommonRec.lambda2);
                    System.out.println(equalStr);

                    fw.write("\n" + equalStr + "\n");
                    fw.write("        lambda3——>" + CommonRec.lambda2 + "\n");
                    fw.write(equalStr + "\n");
                    fw.flush();


                    // 为确保每一次gamma取新值后是一个新的更新过程，则每次重新创建一个MSNLF对象，这些对象的参数取值是一致的
                    ARSL trainSNLF = new ARSL();
//                trainASNLF.printNegativeFeature(); // 检查是否有负初始值
                    // 开始训练
                    trainSNLF.train(metrics, fw);

                    // 输出最优值信息
                    if (metrics == CommonRec.R_Squared) {
                        System.out.println("Max training Result:\t\t\t" + trainSNLF.max_Res);
                        System.out.println("Max total training epochs:\t\t" + trainSNLF.max_Round);
                        System.out.println("Total Round:\t\t" + trainSNLF.total_Round);
                        System.out.println("Max total training time:\t\t" + trainSNLF.maxTotalTime);
                        System.out.println("Max average training time:\t\t" + trainSNLF.maxTotalTime / trainSNLF.max_Round);
                        System.out.println("Total training time:\t\t" + trainSNLF.total_Time);
                        System.out.println("Average training time:\t\t" + trainSNLF.total_Time / trainSNLF.total_Round);

                        fw.write("Max training Result:\t\t\t" + trainSNLF.max_Res + "\n");
                        fw.write("Max total training epochs:\t\t" + trainSNLF.max_Round + "\n");
                        fw.write("Total Round:\t\t" + trainSNLF.total_Round + "\n");
                        fw.write("Max total training time:\t\t" + trainSNLF.maxTotalTime + "\n");
                        fw.write("Max average training time:\t\t" + trainSNLF.maxTotalTime / trainSNLF.max_Round + "\n");
                        fw.write("Total training time:\t\t" + trainSNLF.total_Time + "\n");
                        fw.write("Average training time:\t\t" + trainSNLF.total_Time / trainSNLF.total_Round + "\n");
                    } else {
                        System.out.println("Min training Error:\t\t\t" + trainSNLF.min_Error);
                        System.out.println("Min total training epochs:\t\t" + trainSNLF.min_Round);
                        System.out.println("Total Round:\t\t" + trainSNLF.total_Round);
                        System.out.println("Min total training time:\t\t" + trainSNLF.minTotalTime);
                        System.out.println("Min average training time:\t\t" + trainSNLF.minTotalTime / trainSNLF.min_Round);
                        System.out.println("Total training time:\t\t" + trainSNLF.total_Time);
                        System.out.println("Average training time:\t\t" + trainSNLF.total_Time / trainSNLF.total_Round);

                        fw.write("Min training Error:\t\t\t" + trainSNLF.min_Error + "\n");
                        fw.write("Min total training epochs:\t\t" + trainSNLF.min_Round + "\n");
                        fw.write("Total Round:\t\t" + trainSNLF.total_Round + "\n");
                        fw.write("Min total training time:\t\t" + trainSNLF.minTotalTime + "\n");
                        fw.write("Min average training time:\t\t" + trainSNLF.minTotalTime / trainSNLF.min_Round + "\n");
                        fw.write("Total training time:\t\t" + trainSNLF.total_Time + "\n");
                        fw.write("Average training time:\t\t" + trainSNLF.total_Time / trainSNLF.total_Round + "\n");
                        fw.flush();
                    }


                }

            }

        }
        fw.close();
    }

    public void train(int metrics, FileWriter fw) throws IOException {
        for (RTuple trainR : trainDataSet) {
            double ratingHat = 0;
            for (int dim = 0; dim < featureDimension; dim++) {
                ratingHat += X[trainR.userID][dim] * Y[trainR.itemID][dim];
            }
            trainR.ratingHat = ratingHat;
        }
        double lastErr = 10;
        //int keepRising = 0;
        for (int round = 1; round <= maxRound; round++) {
            double startTime = System.currentTimeMillis();
            for (int dim = 0; dim < featureDimension; dim++) {
                resetAuxArray();
                for (RTuple trainR : trainDataSet) {
                    XU[trainR.userID] += Y[trainR.itemID][dim] * (Z[trainR.userID][trainR.itemID] - trainR.ratingHat + X[trainR.userID][dim] * Y[trainR.itemID][dim]);
                    XD[trainR.userID] += lambda1 * Math.pow(Y[trainR.itemID][dim], 2) + lambda;
                    YU[trainR.itemID] += X[trainR.userID][dim] * (Z[trainR.userID][trainR.itemID] - trainR.ratingHat + X[trainR.userID][dim] * Y[trainR.itemID][dim]);
                    YD[trainR.itemID] += lambda1 * Math.pow(X[trainR.userID][dim], 2) + lambda;
                }
                for (int userID = 1; userID <= maxID; userID++) {
                    XC[userID] = X[userID][dim];
                    YC[userID] = Y[userID][dim];
                    if (XD[userID] != 0) X[userID][dim] = Math.max(0, lambda1 * XU[userID] / XD[userID]);
                    if (YD[userID] != 0) Y[userID][dim] = Math.max(0, lambda1 * YU[userID] / YD[userID]);
                }
                for (RTuple trainR : trainDataSet) {
                    double ratingHatNew = X[trainR.userID][dim] * Y[trainR.itemID][dim] - XC[trainR.userID] * YC[trainR.itemID];
                    trainR.ratingHat = trainR.ratingHat + ratingHatNew;
                }
            }
            for (RTuple trainR : trainDataSet) {
                Z[trainR.userID][trainR.itemID] = (trainR.rating + lambda1 * trainR.ratingHat + lambda2 * Z[trainR.itemID][trainR.userID]) / (1 + lambda1 + lambda2);
            }

            double endTime = System.currentTimeMillis();
            cacheTotalTime += endTime - startTime;
            total_Time += endTime - startTime;

            // 计算本轮训练结束后，在测试集上的误差
            double curErr;
            if (metrics == CommonRec.RMSE) {
                curErr = testRMSE();
            } else {
                curErr = testMAE();
            }
            fw.write(curErr + "\n");
            fw.flush();
            System.out.println(curErr);
            System.out.println(endTime - startTime);

            total_Round += 1;
            if (min_Error > curErr) {
                //keepRising = 0;
                min_Error = curErr;
                min_Round = round;
                this.minTotalTime += this.cacheTotalTime;
                this.cacheTotalTime = 0;
            }else{
                //keepRising++;
            }


            if (Math.abs(curErr - lastErr) > minGap && Math.abs(curErr - lastErr) < maxGap)
                lastErr = curErr;
            else break;
        }

       // outPutModelSym();
//        outPutModelNonnega();
    }
}
