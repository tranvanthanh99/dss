require("dotenv").config();
var mongoose = require("mongoose");
const db = {
    // url : `mongodb://${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_NAME}`,
    // url : `mongodb+srv://admin:admin@cluster0.rwqtf.mongodb.net/FacePlusPlusProject?retryWrites=true&w=majority`,
    url : `mmongodb+srv://admin:admin@cluster0.9hbam.mongodb.net/myFirstDatabase?retryWrites=true&w=majoritys`,
    option : {
        useNewUrlParser : true // de sp mongo cho su dung db cua no
    }
}

mongoose.connect(db.url, db.option);

module.exports = mongoose;

