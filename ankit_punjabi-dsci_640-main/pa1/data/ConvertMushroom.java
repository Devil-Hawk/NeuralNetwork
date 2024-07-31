package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.stream.Collectors;

public class ConvertMushroom {
    public static void main(String[] args) throws IOException {
        String inputFilePath = "./datasets/agaricus-lepiota.data";
        String outputFilePath = "./datasets/agaricus-lepiota-my-output.txt";
        File inputFile = new File(inputFilePath);
        BufferedReader fileReader = new BufferedReader(new FileReader(inputFile));
        BufferedWriter fileWriter = new BufferedWriter(new FileWriter(new File(outputFilePath)));
        String line;
        int numRows = 0;
        while ((line = fileReader.readLine()) != null) {
            numRows++;
        }
        fileReader.close();
        fileReader = new BufferedReader(new FileReader(inputFile));

        String[][] inputData = null;
        int rowCounter = 0;
        while ((line = fileReader.readLine()) != null) {
            String[] valuesForRow = line.split(",");
            inputData = inputData == null ? new String[numRows][valuesForRow.length] : inputData;

            inputData[rowCounter++] = valuesForRow;
        }
        fileReader.close();
        HashMap<Integer, String[]> valueMap = new HashMap<>();
        valueMap.put(0, new String[]{"p"});
        valueMap.put(1, new String[]{"b", "c", "x", "f", "k", "s"});
        valueMap.put(2, new String[]{"f", "g", "y", "s"});
        valueMap.put(3, new String[]{"n", "b", "c", "g", "r", "p", "u", "e", "w", "y"});
        valueMap.put(4, new String[]{"t", "f"});
        valueMap.put(5, new String[]{"a", "l", "c", "y", "f", "m", "n", "p", "s"});
        valueMap.put(6, new String[]{"a", "d", "f", "n"});
        valueMap.put(7, new String[]{"c", "w", "d"});
        valueMap.put(8, new String[]{"b", "n"});
        valueMap.put(9, new String[]{"k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"});
        valueMap.put(10, new String[]{"e", "t"});
        valueMap.put(11, new String[]{"b", "c", "u", "e", "z", "r", "?"});
        valueMap.put(12, new String[]{"f", "y", "k", "s"});
        valueMap.put(13, new String[]{"f", "y", "k", "s"});
        valueMap.put(14, new String[]{"n", "b", "c", "g", "o", "p", "e", "w", "y"});
        valueMap.put(15, new String[]{"n", "b", "c", "g", "o", "p", "e", "w", "y"});
        valueMap.put(16, new String[]{"p", "u"});
        valueMap.put(17, new String[]{"n", "o", "w", "y"});
        valueMap.put(18, new String[]{"n", "o", "t"});
        valueMap.put(19, new String[]{"c", "e", "f", "l", "n", "p", "s", "z"});
        valueMap.put(20, new String[]{"k", "n", "b", "h", "r", "o", "u", "w", "y"});
        valueMap.put(21, new String[]{"a", "c", "n", "s", "v", "y"});
        valueMap.put(22, new String[]{"g", "l", "m", "p", "u", "w", "d"});
        StringBuilder resultStringBuilder = new StringBuilder();
        int currentRow = 0;
        while (currentRow < inputData.length) {
            int currentColumn = 0;
            while (currentColumn < inputData[currentRow].length) {
                String cellValue = inputData[currentRow][currentColumn];
                String[] possibleValues = valueMap.get(currentColumn);
                String binaryRepresentation = Arrays.stream(possibleValues)
                        .map(v -> v.equals(cellValue) ? "1" : "0")
                        .collect(Collectors.joining(","));
                if (currentColumn == 0) {
                    String classValue = binaryRepresentation.split(",")[0];
                    resultStringBuilder.append(classValue).append(":");
                } else if (currentColumn != inputData[currentRow].length - 1) {
                    resultStringBuilder.append(binaryRepresentation).append(",");
                } else {
                    resultStringBuilder.append(binaryRepresentation);
                }
                currentColumn++;
            }
            resultStringBuilder.append("\n");
            currentRow++;
        }
        fileWriter.write(resultStringBuilder.toString());
        fileWriter.close();
    }
}
