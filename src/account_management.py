# AMRUTA OXAI V2
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha

import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import sys, getopt, argparse
import random, string
import pandas as pd
from datetime import datetime
import calendar

def get_db_connection():
    """ connects to user account database """
    return sqlite3.connect('./accounts.db')

def add_account(uname, password, num_months, conn):
    """
    adds a new account with user inputted username and alphanumeric
    generated random password and number of months they have access to
    """
    hash_ = generate_password_hash(password)
    c = conn.cursor()

    # add user name and hashed password
    c.execute("INSERT INTO accounts VALUES (?, ?)", (uname, hash_))

    # add date limits
    today, end_date = get_date_limit(num_months)
    c.execute("INSERT INTO date_limits (user_id, begin_date, end_date) VALUES (?, ?, ?)", (uname, today, end_date))

    print('account for %s added!'%(uname))
    conn.commit()
    conn.close()
    return end_date

def get_date_limit(num_months):
    # today = datetime.date.today()
    today = datetime.now().date()
    end_date = add_months(today, num_months)
    return today, end_date.date()

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)
    # return datetime.date(year, month, day)

def generate_random_password(length=8):
    """
    Generates random 8 character alphanumeric password
    """
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join((random.choice(lettersAndDigits) for i in range(length)))

def show_accounts(conn):
    """ show current accounts """
    df = pd.read_sql_query("SELECT accounts.user_id, begin_date, end_date FROM accounts, date_limits WHERE accounts.user_id=date_limits.user_id", conn)
    if df.shape[1] > 0:
        print()
        print('Existing users')
        # print(df[['user_id', 'begin_date', 'end_date']])
        print(df)
        print()
    else:
        print('no accounts registered')
    conn.close()

def lookup_account(user_nme, conn):
    """ checks if account is available by user name and returns the given account """
    df = pd.read_sql_query("SELECT * FROM accounts", conn)
    if len(df) > 0:
        if user_nme in df['user_id'].values:
            acc = df[df['user_id']==user_nme]
            return acc
        else:
            print('account not found')
            #return None
    else:
        print('no accounts in server')
        #return None

def delete_account(uname, conn):
    """ deletes account """
    c = conn.cursor()
    c.execute("DELETE FROM accounts WHERE user_id=='%s'" % str(uname))
    c.execute("DELETE FROM date_limits WHERE user_id=='%s'" % str(uname))
    conn.commit()
    conn.close()
    print("account for %s deleted." % uname)

def update_access(uname, end_date, conn):
    c = conn.cursor()
    c.execute("UPDATE date_limits SET end_date=(?) WHERE user_id=(?) ", (end_date, uname))
    conn.commit()
    conn.close()
    print("Account access for %s updated until %s" % (uname, str(end_date)))

def validate(uname, pwd):
    """ validate an account by user inputted username and password
        checks username and password by hash
    """
    con = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM accounts", con)
    date_limit_df = pd.read_sql_query("SELECT * FROM date_limits", con)

    if len(df) > 0:
    #     if uname in df['user_id'].values:
    #         print('valid username')
    #         acc = df[df['user_id']==uname]
    #         if check_password_hash(pwhash=acc['password'].values[0], password=pwd):
    #             print('valid password')
    #             return True
    #         else:
    #             return False
    #     else:
    #         return False
    # else:
    #     return False
        if uname in df['user_id'].values:
            print('valid username')
            acc = df[df['user_id'] == uname]
            if check_password_hash(pwhash=acc['password'].values[0], password=pwd):
                print('valid password')
                now = datetime.now()
                date_limit = date_limit_df[date_limit_df['user_id'] == uname]
                if datetime.strptime(date_limit['end_date'].values[0], "%Y-%m-%d").strftime("%Y-%m-%d") > now.strftime(
                        "%Y-%m-%d"):
                    return 'Access Granted'
                else:
                    return 'Subscription Ended'
            else:
                return 'Invalid username/password'
        else:
            return 'Invalid username/password'
    else:
        return 'Invalid username/password'

def validate_access(uname, pwd):
    """
    docstring here
        :param uname:
        :param pwd:
    """
    con = get_db_connection()

    return True


def main():

    ## usage
    parser = argparse.ArgumentParser(description='Amruta XAI Account Management', prog = 'acct_mngmt', usage='%(prog)s [options]')
    parser.add_argument('--show', help='show current user names in the account directory', action="store_true")
    parser.add_argument('--add', help='add a new account', action="store_true")
    parser.add_argument('--delete', help='delete an account', action='store_true')
    parser.add_argument('--update', help='update monthly access limit', action='store_true')
    args = parser.parse_args()

    ## get SQLite database connection
    conn = get_db_connection()

    if args.show:
        show_accounts(conn)
    elif args.add:
        adding_state = True
        while adding_state:
            user_nme = input("Please enter a new user name: ")
            if lookup_account(user_nme, conn) != None:
                print("User name not available.")
            else:
                pwd = generate_random_password()
                num_months = input("Please enter number of months of available access (int): ")
                assert num_months.isnumeric(), 'Must enter an input of type integer.'
                num_months = int(num_months)

                end_date = add_account(user_nme, pwd, num_months, conn)
                print('Account added! \n Username: %s \n Password: %s'%(user_nme, pwd))
                print('Access available until %s'%(str(end_date)))
                adding_state = False
    elif args.delete:
        delete_state = True
        while delete_state:
            user_nme = input("Please enter a user name to delete: ")
            acc = lookup_account(user_nme, conn)
            if len(acc) == 1:
                delete_account(user_nme, conn)
                delete_state = False
    elif args.update:
        update_state = True
        while update_state:
            user_nme = input("Please enter user name to update monthly access: ")
            acc = lookup_account(user_nme, conn)
            if len(acc) == 1:
                num_months = input("Please enter number of months of available access (int): ")
                assert num_months.isnumeric(), 'Must enter an input of type integer.'
                num_months = int(num_months)

                today, end_date = get_date_limit(num_months)
                update_access(user_nme, end_date, conn)
                update_state = False


main()