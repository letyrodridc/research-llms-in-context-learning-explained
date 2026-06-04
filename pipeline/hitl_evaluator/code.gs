// Valid credentials for the research team (Username : Password)
// Change these passwords before deploying for your team.
var VALID_USERS = {
  "leticia": "admin2026",
  "carmen": "icml2026",
  "nacho": "eval2026"
};

// 1. SERVE THE HTML INTERFACE
function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
      .setTitle('Visual HITL Evaluation Dashboard')
      .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

// 2. SETUP THE SPREADSHEET COLUMNS
// Run this function ONCE from the Apps Script editor
function setupSheet() {
  var activeSpreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  if (!activeSpreadsheet) {
    throw new Error("CRITICAL ERROR: Script is not bound to a Google Sheet.");
  }

  var sheet = activeSpreadsheet.getActiveSheet();
  
  if (sheet.getLastColumn() === 0) {
    var headers = [
      "Timestamp", "Evaluator", "Dataset", "Prompt_Type", "Source_Model", "Query_Index",
      "Predicted_Label", "Is_Correct", 
      "TG", "HF", "CC", "CP", "Cn", "S", "LD", "IF", "LC", "Comments"
    ];
    sheet.appendRow(headers);
    sheet.setFrozenRows(1);
    sheet.getRange("A1:R1").setFontWeight("bold");
  }
}

// 3. AUTHENTICATE USER
function authenticate(username, password) {
  var user = (username || "").toLowerCase();
  if (VALID_USERS[user] && VALID_USERS[user] === password) {
    return { status: "success" };
  }
  return { status: "error", message: "Invalid username or password." };
}

// 4. SAVE EVALUATION TO GOOGLE SHEETS
function saveEvaluation(payload) {
  // Double-check authentication on the server side for security
  var auth = authenticate(payload.username, payload.password);
  if (auth.status !== "success") {
    throw new Error("Authentication failed during save operation.");
  }

  var activeSpreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = activeSpreadsheet.getActiveSheet();
  
  var row = [
    new Date(),           
    payload.username,                 
    payload.dataset,         
    payload.promptType,      
    payload.sourceModel,     
    payload.queryIndex,      
    payload.predicted,       
    payload.correct,         
    payload.TG, payload.HF, payload.CC, payload.CP, payload.Cn, payload.S, payload.LD, payload.IF, payload.LC, 
    payload.comments
  ];
  
  sheet.appendRow(row);
  return { status: "success" };
}