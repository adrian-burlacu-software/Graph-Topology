const fs = require("fs").promises;
const csv = require('async-csv');

describe.skip("Generate the syllabic pattern for the 1-7 letter dictionary.", () => {
  let maxId = 0;
  let ids = {};
  let output = [];
  
  const commonSetup = async function () {
	const csvString = await fs.readFile('dictionaries/dictionarySyllables.csv', 'utf-8');
	const rows = await csv.parse(csvString);
	
	for (let row of rows) {
		let pattern = row.slice(2, 14);
		pattern = pattern.map(x => x ? 1 : -1);
		if (typeof ids[pattern] !== 'undefined') { continue; }
		
		ids[pattern] = maxId++;
	}
	
	output.push(["word", "length", "patternId"]);
	for (let row of rows) {
		let pattern = row.slice(2, 14);
		pattern = pattern.map(x => x ? 1 : -1);
		output.push([row[0], row[1], ids[pattern]]);
	}
  };
  
  // Writing the network to disk
  const commonProcessing = async function () {    
    await fs.writeFile(`results/dictionaryPatterns.csv`, await csv.stringify(output),'utf-8');
  }
 
  test("Generating the syllabic pattern for the 1-7 letter dictionary.", async () => {
    await commonSetup();
    await commonProcessing();
	
    // Automatically passing tests
    expect(true).toEqual(true);
  });
});
