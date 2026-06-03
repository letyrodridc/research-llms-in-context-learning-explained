// Sirve la interfaz HTML al usuario
function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
      .setTitle('Evaluación Humana XAI (HITL)')
      .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

// Ejecuta esto una sola vez desde el editor para preparar las columnas
function setupSheet() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  if (sheet.getLastColumn() === 0) {
    var headers = [
      "Fecha", "Archivo", "Trial", "Clase_Esperada", "Correcto", 
      "TG", "HF", "CC", "CP", "Cn", "S", "LD", "IF", "LC", "Comentarios"
    ];
    sheet.appendRow(headers);
    // Congelar la primera fila
    sheet.setFrozenRows(1);
    sheet.getRange("A1:O1").setFontWeight("bold");
  }
}

// Guarda la evaluación individual enviada desde el navegador
function saveEvaluation(data) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  
  // Prepara la fila con todos los datos
  var row = [
    new Date(),           // Fecha y hora exacta
    data.fileName,        // Nombre del archivo de log
    data.trial,           // El identificador (ej: Run 0 | Query 1)
    data.expected,        // Clase esperada
    data.correct,         // Si el modelo acertó o no
    data.TG, 
    data.HF, 
    data.CC, 
    data.CP, 
    data.Cn, 
    data.S, 
    data.LD, 
    data.IF, 
    data.LC, 
    data.comments
  ];
  
  // Agrega la fila al final del documento
  sheet.appendRow(row);
  return true; // Confirma que se guardó
}