import requests


def subscribe_to_api(api_url, subscription_key):
    headers = {
        "X-RapidAPI-Host": "api.rapidapi.com",
        "X-RapidAPI-Key": subscription_key
    }

    response = requests.post(api_url, headers=headers)

    if response.status_code == 200:
        return True, "订阅成功！"
    else:
        return False, f"订阅失败。错误信息: {response.text}"


def main():
    # api_url = input("请输入要订阅的 API 的 URL: ")
    # subscription_key = input("请输入你的 RapidAPI 订阅密钥: ")

    api_url = 'https://checkmail1.p.rapidapi.com/'
    subscription_key = 'f2a21f6d95msh788d0da7bdf3129p1b38ffjsnf2da952138d6'
    success, message = subscribe_to_api(api_url, subscription_key)

    if success:
        print(message)
    else:
        print(message)


if __name__ == "__main__":
    main()

# https://0007.p.rapidapi.com/
# f2a21f6d95msh788d0da7bdf3129p1b38ffjsnf2da952138d6