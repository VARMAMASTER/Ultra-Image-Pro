from app import app

if __name__ == '__main__':
    app.debug = True
    # app.run(host="localhost",port=5070)
    app.run(debug=True)