import hex.genmodel.GenModel;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.exception.PredictException;
import hex.genmodel.easy.prediction.BinomialModelPrediction;
import hex.genmodel.easy.prediction.MultinomialModelPrediction;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.BufferedWriter;
import java.io.OutputStreamWriter;

public class Main {

	public static void main(String[] args) {
		try {
 
            InputStreamReader in= new InputStreamReader(System.in);
 
            BufferedReader input = new BufferedReader(in);
 
            String str;

			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out));
			out.write("Begin:\n");
			out.flush();
 
            while ((str = input.readLine()) != null) {
				if(str.equals("End")) {
					out.write(str + '\n');
					out.flush();
					out.close();
					return;
				}
				
				String result;
				try {
					result = getResult(str.split(" "));
				}
				catch (PredictException pE) {
					result = "ERROR";
				}
				
				out.write(result + '\n');
				out.flush();
            }
 
        } catch (IOException io) {
            io.printStackTrace();
        }
	}
	
	private static String getResult(String[] args) throws PredictException {
		hex.genmodel.GenModel rawModel = getClassInstance(args[0]);
	    EasyPredictModelWrapper model = new EasyPredictModelWrapper(rawModel);
	    
	    RowData row = new RowData();
	    String[] columnLabels = rawModel.getNames();

	    for (int i = 0; i < columnLabels.length; i++) {
			row.put(columnLabels[i], args[i+1]);
		}

	    if (model.getModelCategory() == hex.ModelCategory.Binomial) {
			BinomialModelPrediction p = model.predictBinomial(row);
			return p.label;
		}
	    else {
	    	MultinomialModelPrediction p = model.predictMultinomial(row);
			return p.label;
	    }	    
	}
	
	private static GenModel getClassInstance(String key) {
		switch (key) {
			case "TrieEnd3":
				return new TrieEnd3();
			case "TrieEnd40":
				return new TrieEnd40();
			case "TrieEnd41":
				return new TrieEnd41();
			case "TrieEnd42":
				return new TrieEnd42();
			case "TrieEndSwitch":
				return new TrieEndSwitch();
			case "TrieCheck3":
				return new TrieCheck3();
			case "TrieCheck40":
				return new TrieCheck40();
			case "TrieCheck41":
				return new TrieCheck41();
			case "TrieCheck42":
				return new TrieCheck42();
			case "TrieModel3":
				return new TrieModel3();
			case "TrieModel40":
				return new TrieModel40();
			case "TrieModel41":
				return new TrieModel41();
			case "TrieModel42":
				return new TrieModel42();
			case "TrieModelSwitch":
				return new TrieModelSwitch();
			case "LoopMethod":
				return new LoopMethod();
			case "LoopSequence":
				return new LoopSequence();
			case "SearchArg":
				return new SearchArg();
			case "SearchLoopArg":
				return new SearchLoopArg();
			case "SearchLoopMethod":
				return new SearchLoopMethod();
			case "SearchLoopResult":
				return new SearchLoopResult();
			case "SearchLoopSequence":
				return new SearchLoopSequence();
			case "SearchMethod":
				return new SearchMethod();
			case "SearchResult":
				return new SearchResult();
			case "SearchSequence":
				return new SearchSequence();
			default:
				return null;
		}
	}
}
