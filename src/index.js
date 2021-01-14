let execShellCommand = require("./execShellCommand.js");
let prompt;

async function Main () {
  console.log("1. Building.");
  let stdout = await execShellCommand('npm install');
  console.log(stdout);
  console.log("");
  prompt = require("prompt-async");

  console.log("2. You need to enable the following commands: java(Orace JRE) and npm(NodeJS) on your Windows system.");
  console.log("");
  console.log("If you want to re-build using the portable JDK or you want help changing  your environment variable,");
  console.log("You can install a portable version of Java: https://portapps.io/download/oracle-jdk-portable-win64-11.0.7-17-setup.exe");
  console.log("Run oracle-jdk-portable.exe from the oracle-jdk-portable folder.");
  console.log("You need to set your own environment letiable for this portable NodeJS.");
  console.log("https://github.com/garethflowers/nodejs-portable/releases/download/v4.2.1/NodeJSPortable_4.2.1.paf.exe");
  console.log("");

  console.log("This is a global setup script should this project need it.");
  console.log("Any changes will be applied before the tests are run with 'npm test'.");
  console.log("You only need to run 'npm start' once, otherwise use 'npm test' continue? (y/yes)");
  console.log("");
  prompt.start();
  const {start} = await prompt.get(["start"]);
  if (!(start === "y" || start === "yes")) {
    return;
  }

  console.log("");
  console.log("Do you want to run javac to rebuild the Java project? (y/yes)");
  prompt.start();
  const {javac} = await prompt.get(["javac"]);
  if (javac === "y" || javac === "yes") {
    //stdout = await execShellCommand('for /r src %i in (*.class) do del %i');
    stdout = await execShellCommand('javac -cp "src;src\\h2o-genmodel.jar" src\\Main.java -d src');
  }
  console.log(stdout);
  console.log("");

  console.log("3. Creating results folder.");
  stdout = await execShellCommand('if not exist results mkdir results');
  console.log(stdout);

  console.log("4. Running 'npm test'.");
  stdout = await execShellCommand('npm test');
  console.log(stdout);

  console.log("You can check the Coverage and Results folders for output.");
  console.log("If you want to re-run in debug mode, you can manually delete the files in the results folder and then run:");
  console.log("  node --max-old-space-size=6144 --inspect-brk ./node_modules/jest/bin/jest.js --runInBand --testTimeout=300000");
  console.log("  There is an extension that connects to the debug session from the browser:");
  console.log("  https://chrome.google.com/webstore/detail/nodejs-v8-inspector-manag/gnhhdgbaldcilmgcpfddgdbkhjohddkj?hl=en");
  console.log("Thank you! ALL your code will belong to us ;).");
};
Main();