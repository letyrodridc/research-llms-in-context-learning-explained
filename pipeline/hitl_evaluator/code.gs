// Valid credentials for the research team (Username : Password)
// Change these passwords before deploying for your team.
var VALID_USERS = {
  "leticia": "admin2026",
  "carmen": "icml2026",
  "nacho": "eval2026"
};

// Run this function ONCE from the Apps Script editor to set up the spreadsheet columns
function setupSheet() {
  var activeSpreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  
  // FAIL-SAFE: Check if the script is properly bound to a Google Sheet
  if (!activeSpreadsheet) {
    throw new Error("CRITICAL ERROR: This script is not attached to a Google Sheet. You must open your Google Sheet, click 'Extensions' > 'Apps Script', and paste this code there. Do not create the script from script.google.com directly.");
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

// Handles incoming POST requests (Both Login and Save Evaluation)
function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    
    // 1. AUTHENTICATION CHECK
    var user = (data.username || "").toLowerCase();
    var pass = data.password || "";
    
    if (!VALID_USERS[user] || VALID_USERS[user] !== pass) {
      return ContentService.createTextOutput(JSON.stringify({
          "status": "error", 
          "message": "Authentication failed. Invalid username or password."
        }))
        .setMimeType(ContentService.MimeType.JSON)
        .setHeader("Access-Control-Allow-Origin", "*");
    }

    // 2. ACTION: LOGIN (Just checking credentials)
    if (data.action === "login") {
      return ContentService.createTextOutput(JSON.stringify({
          "status": "success", 
          "message": "Login successful."
        }))
        .setMimeType(ContentService.MimeType.JSON)
        .setHeader("Access-Control-Allow-Origin", "*");
    }

    // 3. ACTION: SAVE EVALUATION
    if (data.action === "save") {
      var activeSpreadsheet = SpreadsheetApp.getActiveSpreadsheet();
      if (!activeSpreadsheet) {
        throw new Error("Server Error: Script is not bound to a spreadsheet.");
      }
      var sheet = activeSpreadsheet.getActiveSheet();
      
      var row = [
        new Date(),           
        user,                 
        data.dataset,         
        data.promptType,      
        data.sourceModel,     
        data.queryIndex,      
        data.predicted,       
        data.correct,         
        data.TG, data.HF, data.CC, data.CP, data.Cn, data.S, data.LD, data.IF, data.LC, 
        data.comments
      ];
      
      sheet.appendRow(row);
      
      return ContentService.createTextOutput(JSON.stringify({"status": "success"}))
        .setMimeType(ContentService.MimeType.JSON)
        .setHeader("Access-Control-Allow-Origin", "*");
    }
      
  } catch(error) {
    return ContentService.createTextOutput(JSON.stringify({"status": "error", "message": error.toString()}))
      .setMimeType(ContentService.MimeType.JSON)
      .setHeader("Access-Control-Allow-Origin", "*");
  }
}

// Handles preflight CORS requests from the browser
function doOptions(e) {
  var headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400"
  };
  var response = ContentService.createTextOutput();
  for (var key in headers) {
    response.setHeader(key, headers[key]);
  }
  return response;
}