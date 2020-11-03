const fs = require("fs").promises;
const csv = require('async-csv');
let TrieSymbolicGenerator = require("../src/trieSymbolicGenerator");

describe.skip("Generate the training data for the ML model.", () => {
  // Getting the common steps done with arguments.
  const commonSetup = async function (numLetters) {
    let trieSymbolicGenerator = new TrieSymbolicGenerator();
    
    // Read file from disk:
    const csvString = await fs.readFile('dictionaries/dictionaryN4.csv', 'utf-8');
    const rows = await csv.parse(csvString);
    
    for (let word of rows) {
      if (word[0].length > numLetters) {
        continue;
      }
      trieSymbolicGenerator.insert(word[0]);
    }
    
    return trieSymbolicGenerator;
  };
  
  // Writing the network to disk
  const commonProcessing = async function (numLetters, trieSymbolicGenerator) {
    const output = [];
    output.push(["parentNodeIdIn", "nodeIndexIn","nodeIdOut","isEndOfWordOut"]);

    for (i = 0; i < trieSymbolicGenerator.trieStats.nodes.length; i++) {
      let node = trieSymbolicGenerator.trieStats.nodes[i];
      output.push([node.parent?node.parent.Id:"null", node.index == null ? "null" : node.index, node.Id, node.isEndOfWord ? "true":"false"]);
    };
    
    await fs.writeFile(`results/TrieModel_${numLetters}_.csv`, await csv.stringify(output),'utf-8');
  }
 
  test("Generating the graph of words with three letters.", async () => {
    const numLetters = 3;

    let trieSymbolicGenerator = await commonSetup(numLetters);
    await commonProcessing(numLetters, trieSymbolicGenerator);
    
    // Automatically passing tests
    expect(true).toEqual(true);
  });
  
  test("Generating the graph of words with four letters.", async () => {
    const numLetters = 4;
    let trieSymbolicGenerator = await commonSetup(numLetters);
    await commonProcessing(numLetters, trieSymbolicGenerator);
    
    // Automatically passing tests
    expect(true).toEqual(true);
  });
});
